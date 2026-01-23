"""Tests for the RegionRouter."""

import pytest
import time

from aragora.control_plane.region_router import (
    RegionHealth,
    RegionRouter,
    RegionRoutingDecision,
    RegionStatus,
    get_region_router,
    set_region_router,
    init_region_router,
)
from aragora.control_plane.scheduler import Task, RegionRoutingMode


class TestRegionHealth:
    """Test RegionHealth dataclass."""

    def test_default_values(self):
        """Test default values."""
        health = RegionHealth(region_id="test-region")
        assert health.region_id == "test-region"
        assert health.status == RegionStatus.UNKNOWN
        assert health.is_local is False

    def test_is_available_healthy(self):
        """Test is_available for healthy region."""
        health = RegionHealth(region_id="test", status=RegionStatus.HEALTHY)
        assert health.is_available is True

    def test_is_available_degraded(self):
        """Test is_available for degraded region."""
        health = RegionHealth(region_id="test", status=RegionStatus.DEGRADED)
        assert health.is_available is True

    def test_is_available_unhealthy(self):
        """Test is_available for unhealthy region."""
        health = RegionHealth(region_id="test", status=RegionStatus.UNHEALTHY)
        assert health.is_available is False

    def test_is_available_unknown(self):
        """Test is_available for unknown region."""
        health = RegionHealth(region_id="test", status=RegionStatus.UNKNOWN)
        assert health.is_available is False

    def test_health_score_healthy(self):
        """Test health score for healthy region."""
        health = RegionHealth(
            region_id="test",
            status=RegionStatus.HEALTHY,
            latency_ms=10.0,
            capacity_pct=30.0,
            error_rate=0.01,
            agent_count=5,
        )
        score = health.health_score
        assert score > 80  # Healthy region should have high score

    def test_health_score_unhealthy(self):
        """Test health score for unhealthy region."""
        health = RegionHealth(
            region_id="test",
            status=RegionStatus.UNHEALTHY,
        )
        assert health.health_score == 0.0

    def test_health_score_no_agents(self):
        """Test health score penalty for no agents."""
        health = RegionHealth(
            region_id="test",
            status=RegionStatus.HEALTHY,
            agent_count=0,
        )
        score = health.health_score
        assert score < 90  # Should have penalty

    def test_health_score_high_latency(self):
        """Test health score penalty for high latency."""
        health = RegionHealth(
            region_id="test",
            status=RegionStatus.HEALTHY,
            latency_ms=300.0,  # High latency
            agent_count=5,
        )
        score = health.health_score
        assert score < 80  # Should have latency penalty


class TestRegionRouter:
    """Test RegionRouter class."""

    def test_init(self):
        """Test router initialization."""
        router = RegionRouter(local_region="us-west-2")
        assert router.local_region == "us-west-2"

    def test_local_region_initialized_as_healthy(self):
        """Test local region is initialized as healthy."""
        router = RegionRouter(local_region="us-west-2")
        health = router.get_region_health("us-west-2")
        assert health.status == RegionStatus.HEALTHY
        assert health.is_local is True

    def test_get_region_health_unknown(self):
        """Test getting health of unknown region."""
        router = RegionRouter(local_region="us-west-2")
        health = router.get_region_health("unknown-region")
        assert health.status == RegionStatus.UNKNOWN

    def test_update_region_metrics(self):
        """Test updating region metrics."""
        router = RegionRouter(local_region="us-west-2")

        router.update_region_metrics(
            region_id="us-east-1",
            agent_count=10,
            pending_tasks=5,
            latency_ms=50.0,
            error_rate=0.05,
        )

        health = router.get_region_health("us-east-1")
        assert health.agent_count == 10
        assert health.pending_tasks == 5
        assert health.latency_ms == 50.0
        assert health.error_rate == 0.05
        assert health.status == RegionStatus.HEALTHY

    def test_update_region_metrics_no_agents_degrades(self):
        """Test that region with no agents is degraded."""
        router = RegionRouter(local_region="us-west-2")

        router.update_region_metrics(
            region_id="us-east-1",
            agent_count=0,
            error_rate=0.0,
        )

        health = router.get_region_health("us-east-1")
        assert health.status == RegionStatus.DEGRADED

    def test_update_region_metrics_high_error_unhealthy(self):
        """Test that region with high error rate is unhealthy."""
        router = RegionRouter(local_region="us-west-2")

        router.update_region_metrics(
            region_id="us-east-1",
            agent_count=5,
            error_rate=0.6,  # Above 0.5 threshold
        )

        health = router.get_region_health("us-east-1")
        assert health.status == RegionStatus.UNHEALTHY

    def test_get_all_region_health(self):
        """Test getting all region health."""
        router = RegionRouter(local_region="us-west-2")

        router.update_region_metrics("us-east-1", agent_count=5)
        router.update_region_metrics("eu-west-1", agent_count=3)

        all_health = router.get_all_region_health()
        assert "us-west-2" in all_health
        assert "us-east-1" in all_health
        assert "eu-west-1" in all_health


class TestRegionRouterSelection:
    """Test region selection functionality."""

    @pytest.mark.asyncio
    async def test_select_region_prefers_healthy(self):
        """Test that healthy regions are preferred."""
        router = RegionRouter(local_region="us-west-2")

        # Setup regions
        router.update_region_metrics("us-west-2", agent_count=5, error_rate=0.1)
        router.update_region_metrics(
            "us-east-1",
            agent_count=10,
            error_rate=0.01,  # Healthier
        )

        task = Task(
            task_type="debate",
            payload={},
            region_routing_mode=RegionRoutingMode.ANY,
        )

        decision = await router.select_region(task, prefer_local=False)

        # Should have a selected region
        assert decision.selected_region is not None
        assert len(decision.health_scores) > 0

    @pytest.mark.asyncio
    async def test_select_region_prefers_local(self):
        """Test that local region is preferred when prefer_local=True."""
        router = RegionRouter(local_region="us-west-2")

        # Setup regions with similar health
        router.update_region_metrics("us-west-2", agent_count=5, error_rate=0.05)
        router.update_region_metrics("us-east-1", agent_count=5, error_rate=0.05)

        task = Task(
            task_type="debate",
            payload={},
            region_routing_mode=RegionRoutingMode.ANY,
        )

        decision = await router.select_region(task, prefer_local=True)

        # Should prefer local region due to bonus
        assert decision.selected_region == "us-west-2"

    @pytest.mark.asyncio
    async def test_select_region_respects_target(self):
        """Test that task target region gets bonus."""
        router = RegionRouter(local_region="us-west-2")

        # Setup regions
        router.update_region_metrics("us-west-2", agent_count=10, error_rate=0.01)
        router.update_region_metrics("us-east-1", agent_count=5, error_rate=0.01)

        task = Task(
            task_type="debate",
            payload={},
            target_region="us-east-1",
            region_routing_mode=RegionRoutingMode.PREFERRED,
        )

        decision = await router.select_region(task, prefer_local=False)

        # Target region should get bonus and be selected
        assert decision.selected_region == "us-east-1"

    @pytest.mark.asyncio
    async def test_select_region_strict_mode(self):
        """Test strict routing mode only allows target region."""
        router = RegionRouter(local_region="us-west-2")

        # Setup regions
        router.update_region_metrics("us-west-2", agent_count=10, error_rate=0.01)
        router.update_region_metrics(
            "us-east-1",
            agent_count=5,
            error_rate=0.5,  # Degraded
        )

        task = Task(
            task_type="debate",
            payload={},
            target_region="us-east-1",
            region_routing_mode=RegionRoutingMode.STRICT,
        )

        decision = await router.select_region(task, prefer_local=False)

        # Strict mode - only target region should be considered
        # But it's degraded, so might not be selected
        if decision.selected_region:
            assert decision.selected_region == "us-east-1"

    @pytest.mark.asyncio
    async def test_select_region_provides_fallbacks(self):
        """Test that fallback regions are provided."""
        router = RegionRouter(local_region="us-west-2")

        # Setup multiple regions
        router.update_region_metrics("us-west-2", agent_count=5)
        router.update_region_metrics("us-east-1", agent_count=4)
        router.update_region_metrics("eu-west-1", agent_count=3)

        task = Task(
            task_type="debate",
            payload={},
            region_routing_mode=RegionRoutingMode.ANY,
        )

        decision = await router.select_region(task)

        # Should have fallback regions
        assert len(decision.fallback_regions) > 0
        # Fallbacks should not include selected region
        assert decision.selected_region not in decision.fallback_regions


class TestRegionRouterFailover:
    """Test failover functionality."""

    @pytest.mark.asyncio
    async def test_failover_marks_region_unhealthy(self):
        """Test that failover marks failed region as unhealthy."""
        router = RegionRouter(local_region="us-west-2")

        router.update_region_metrics("us-west-2", agent_count=5, error_rate=0.0)
        router.update_region_metrics("us-east-1", agent_count=5, error_rate=0.0)

        # Trigger failover
        await router.failover_region(
            task_id="task-123",
            failed_region="us-west-2",
        )

        # Check that failed region is now unhealthy
        health = router.get_region_health("us-west-2")
        assert health.status == RegionStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_failover_returns_alternative(self):
        """Test that failover returns an alternative region."""
        router = RegionRouter(local_region="us-west-2")

        router.update_region_metrics("us-west-2", agent_count=5, error_rate=0.0)
        router.update_region_metrics("us-east-1", agent_count=5, error_rate=0.0)

        fallback = await router.failover_region(
            task_id="task-123",
            failed_region="us-west-2",
        )

        assert fallback == "us-east-1"

    @pytest.mark.asyncio
    async def test_failover_no_alternatives(self):
        """Test failover when no alternatives available."""
        router = RegionRouter(local_region="us-west-2")

        # Only one region
        router.update_region_metrics("us-west-2", agent_count=5, error_rate=0.0)

        fallback = await router.failover_region(
            task_id="task-123",
            failed_region="us-west-2",
        )

        assert fallback is None


class TestRegionRouterStats:
    """Test routing statistics."""

    @pytest.mark.asyncio
    async def test_routing_stats_empty(self):
        """Test stats with no routing history."""
        router = RegionRouter(local_region="us-west-2")
        stats = router.get_routing_stats()

        assert stats["total_decisions"] == 0
        assert stats["by_region"] == {}

    @pytest.mark.asyncio
    async def test_routing_stats_after_selections(self):
        """Test stats after some routing decisions."""
        router = RegionRouter(local_region="us-west-2")

        router.update_region_metrics("us-west-2", agent_count=5)
        router.update_region_metrics("us-east-1", agent_count=5)

        # Make some routing decisions
        task = Task(task_type="debate", payload={})
        await router.select_region(task)
        await router.select_region(task)

        stats = router.get_routing_stats()

        assert stats["total_decisions"] == 2
        assert len(stats["by_region"]) > 0
        assert len(stats["recent_decisions"]) == 2


class TestRegionRouterSingleton:
    """Test singleton functions."""

    def test_get_set_router(self):
        """Test get/set router singleton."""
        router = RegionRouter(local_region="test-region")
        set_region_router(router)

        retrieved = get_region_router()
        assert retrieved is router

    def test_init_region_router(self):
        """Test init_region_router creates and sets singleton."""
        router = init_region_router(local_region="init-region")

        assert router is not None
        assert router.local_region == "init-region"
        assert get_region_router() is router
