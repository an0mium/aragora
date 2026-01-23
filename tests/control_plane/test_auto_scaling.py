"""Tests for Agent Auto-Scaling."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.auto_scaling import (
    AutoScaler,
    ScalingDecision,
    ScalingDirection,
    ScalingMetrics,
    ScalingPolicy,
    ScalingReason,
    get_auto_scaler,
    init_auto_scaler,
    set_auto_scaler,
)


class TestScalingPolicy:
    """Test ScalingPolicy configuration."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = ScalingPolicy.default()

        assert policy.queue_depth_threshold == 10
        assert policy.min_agents == 1
        assert policy.max_agents == 20
        assert policy.scale_up_increment == 2

    def test_aggressive_policy(self):
        """Test aggressive policy has lower thresholds."""
        policy = ScalingPolicy.aggressive()

        assert policy.queue_depth_threshold < ScalingPolicy.default().queue_depth_threshold
        assert policy.cooldown_seconds < ScalingPolicy.default().cooldown_seconds

    def test_conservative_policy(self):
        """Test conservative policy has higher thresholds."""
        policy = ScalingPolicy.conservative()

        assert policy.queue_depth_threshold > ScalingPolicy.default().queue_depth_threshold
        assert policy.cooldown_seconds > ScalingPolicy.default().cooldown_seconds

    def test_provider_limits(self):
        """Test provider-specific limits are defined."""
        policy = ScalingPolicy.default()

        assert "anthropic" in policy.max_agents_per_provider
        assert "openai" in policy.max_agents_per_provider
        assert policy.max_agents_per_provider["anthropic"] >= 5


class TestScalingMetrics:
    """Test ScalingMetrics calculations."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = ScalingMetrics()

        assert metrics.queue_depth == 0
        assert metrics.total_agents == 0
        assert metrics.utilization == 0.0

    def test_agent_exhaustion_ratio_no_agents(self):
        """Test exhaustion ratio is 1.0 when no agents."""
        metrics = ScalingMetrics(total_agents=0, busy_agents=0)

        assert metrics.agent_exhaustion_ratio == 1.0

    def test_agent_exhaustion_ratio_partial(self):
        """Test exhaustion ratio calculation."""
        metrics = ScalingMetrics(total_agents=10, busy_agents=5)

        assert metrics.agent_exhaustion_ratio == 0.5

    def test_agent_exhaustion_ratio_full(self):
        """Test exhaustion ratio when all agents busy."""
        metrics = ScalingMetrics(total_agents=10, busy_agents=10)

        assert metrics.agent_exhaustion_ratio == 1.0


class TestScalingDecision:
    """Test ScalingDecision properties."""

    def test_should_scale_up(self):
        """Test should_scale returns True for scale-up."""
        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_UP,
            reason=ScalingReason.QUEUE_DEPTH,
            recommended_delta=2,
        )

        assert decision.should_scale is True

    def test_should_scale_down(self):
        """Test should_scale returns True for scale-down."""
        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_DOWN,
            reason=ScalingReason.IDLE_AGENTS,
            recommended_delta=-1,
        )

        assert decision.should_scale is True

    def test_should_not_scale_none(self):
        """Test should_scale returns False for no action."""
        decision = ScalingDecision(
            direction=ScalingDirection.NONE,
            reason=ScalingReason.MANUAL,
            recommended_delta=0,
        )

        assert decision.should_scale is False

    def test_should_not_scale_zero_delta(self):
        """Test should_scale returns False when delta is zero."""
        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_UP,
            reason=ScalingReason.QUEUE_DEPTH,
            recommended_delta=0,
        )

        assert decision.should_scale is False


class TestAutoScaler:
    """Test AutoScaler functionality."""

    def test_init_default_policy(self):
        """Test initialization with default policy."""
        scaler = AutoScaler()

        assert scaler.policy is not None
        assert scaler.policy.min_agents == 1

    def test_init_custom_policy(self):
        """Test initialization with custom policy."""
        policy = ScalingPolicy(min_agents=5, max_agents=50)
        scaler = AutoScaler(policy=policy)

        assert scaler.policy.min_agents == 5
        assert scaler.policy.max_agents == 50

    def test_cooldown_passed_initially(self):
        """Test cooldown passes on fresh scaler."""
        scaler = AutoScaler()

        assert scaler._cooldown_passed() is True

    def test_cooldown_not_passed_after_scale(self):
        """Test cooldown doesn't pass immediately after scaling."""
        scaler = AutoScaler()
        scaler._last_scale_time = time.time()

        assert scaler._cooldown_passed() is False

    def test_record_latency(self):
        """Test latency recording."""
        scaler = AutoScaler()

        scaler.record_latency(100.0)
        scaler.record_latency(200.0)
        scaler.record_latency(300.0)

        assert len(scaler._latency_samples) == 3

    def test_record_latency_trimming(self):
        """Test latency samples are trimmed."""
        scaler = AutoScaler()
        scaler._max_latency_samples = 5

        for i in range(10):
            scaler.record_latency(float(i * 100))

        assert len(scaler._latency_samples) == 5
        assert scaler._latency_samples[0] == 500.0  # Oldest kept

    def test_get_stats(self):
        """Test statistics retrieval."""
        scaler = AutoScaler()
        stats = scaler.get_stats()

        assert "running" in stats
        assert "policy" in stats
        assert "scaling_history" in stats

    def test_set_policy(self):
        """Test policy update."""
        scaler = AutoScaler()
        new_policy = ScalingPolicy(min_agents=10)

        scaler.set_policy(new_policy)

        assert scaler.policy.min_agents == 10


class TestAutoScalerScaleUpDecisions:
    """Test scale-up decision logic."""

    @pytest.mark.asyncio
    async def test_scale_up_on_queue_depth(self):
        """Test scale-up triggered by queue depth."""
        policy = ScalingPolicy(queue_depth_threshold=5, max_agents=10)
        scaler = AutoScaler(policy=policy)

        # Mock metrics collection to return high queue depth
        metrics = ScalingMetrics(
            queue_depth=10,
            total_agents=2,
            available_agents=0,
            busy_agents=2,
        )

        with patch.object(scaler, "_collect_metrics", return_value=metrics):
            decision = await scaler.evaluate()

        assert decision.direction == ScalingDirection.SCALE_UP
        assert decision.reason == ScalingReason.AGENT_EXHAUSTION  # Exhaustion checked first
        assert decision.should_scale is True

    @pytest.mark.asyncio
    async def test_scale_up_on_agent_exhaustion(self):
        """Test scale-up triggered by agent exhaustion."""
        policy = ScalingPolicy(agent_exhaustion_threshold=0.8, max_agents=10)
        scaler = AutoScaler(policy=policy)

        metrics = ScalingMetrics(
            queue_depth=2,
            total_agents=5,
            busy_agents=5,  # 100% exhausted
            available_agents=0,
        )

        with patch.object(scaler, "_collect_metrics", return_value=metrics):
            decision = await scaler.evaluate()

        assert decision.direction == ScalingDirection.SCALE_UP
        assert decision.reason == ScalingReason.AGENT_EXHAUSTION

    @pytest.mark.asyncio
    async def test_no_scale_up_at_max_capacity(self):
        """Test no scale-up when at max capacity."""
        policy = ScalingPolicy(
            max_agents=5,
            min_agents=5,  # Also at min to prevent scale-down
            min_utilization_for_scale_down=0.0,  # Disable scale-down
        )
        scaler = AutoScaler(policy=policy)

        metrics = ScalingMetrics(
            queue_depth=100,  # High queue
            total_agents=5,  # At max
            busy_agents=5,
            utilization=1.0,  # High utilization to prevent scale-down
        )

        with patch.object(scaler, "_collect_metrics", return_value=metrics):
            decision = await scaler.evaluate()

        # Scale-up check returns "at max capacity" but then scale-down is checked
        # With utilization at 1.0, scale-down won't trigger either
        assert decision.direction == ScalingDirection.NONE


class TestAutoScalerScaleDownDecisions:
    """Test scale-down decision logic."""

    @pytest.mark.asyncio
    async def test_scale_down_on_idle(self):
        """Test scale-down triggered by idle agents."""
        policy = ScalingPolicy(
            queue_empty_duration_seconds=60,
            min_agents=1,
        )
        scaler = AutoScaler(policy=policy)
        scaler._queue_empty_since = time.time() - 120  # Empty for 2 minutes

        metrics = ScalingMetrics(
            queue_depth=0,
            total_agents=5,
            busy_agents=0,
            queue_empty_duration_seconds=120,
        )

        with patch.object(scaler, "_collect_metrics", return_value=metrics):
            decision = await scaler.evaluate()

        assert decision.direction == ScalingDirection.SCALE_DOWN
        assert decision.reason == ScalingReason.IDLE_AGENTS

    @pytest.mark.asyncio
    async def test_scale_down_on_low_utilization(self):
        """Test scale-down triggered by low utilization."""
        policy = ScalingPolicy(
            min_utilization_for_scale_down=0.3,
            min_agents=1,
        )
        scaler = AutoScaler(policy=policy)

        metrics = ScalingMetrics(
            queue_depth=1,
            total_agents=10,
            busy_agents=1,
            utilization=0.1,  # 10% utilization
        )

        with patch.object(scaler, "_collect_metrics", return_value=metrics):
            decision = await scaler.evaluate()

        assert decision.direction == ScalingDirection.SCALE_DOWN
        assert decision.reason == ScalingReason.COST_OPTIMIZATION

    @pytest.mark.asyncio
    async def test_no_scale_down_at_min_capacity(self):
        """Test no scale-down when at min capacity."""
        policy = ScalingPolicy(min_agents=2)
        scaler = AutoScaler(policy=policy)

        metrics = ScalingMetrics(
            queue_depth=0,
            total_agents=2,  # At min
            busy_agents=0,
            queue_empty_duration_seconds=9999,
        )

        with patch.object(scaler, "_collect_metrics", return_value=metrics):
            decision = await scaler.evaluate()

        # Should not scale down when at minimum capacity
        assert decision.direction == ScalingDirection.NONE
        assert decision.recommended_delta == 0


class TestAutoScalerProviderSelection:
    """Test provider selection for scaling."""

    def test_select_provider_for_scale_up(self):
        """Test provider selection respects priority."""
        policy = ScalingPolicy(
            scale_up_provider_priority=["anthropic", "openai"],
            max_agents_per_provider={"anthropic": 5, "openai": 5},
        )
        scaler = AutoScaler(policy=policy)

        metrics = ScalingMetrics(
            agents_by_provider={"anthropic": 2, "openai": 3},
        )

        provider = scaler._select_provider_for_scale_up(metrics)

        assert provider == "anthropic"  # First in priority with capacity

    def test_select_provider_for_scale_up_skip_full(self):
        """Test provider selection skips full providers."""
        policy = ScalingPolicy(
            scale_up_provider_priority=["anthropic", "openai"],
            max_agents_per_provider={"anthropic": 5, "openai": 5},
        )
        scaler = AutoScaler(policy=policy)

        metrics = ScalingMetrics(
            agents_by_provider={"anthropic": 5, "openai": 3},  # Anthropic full
        )

        provider = scaler._select_provider_for_scale_up(metrics)

        assert provider == "openai"

    def test_select_provider_for_scale_down(self):
        """Test provider selection for scale-down."""
        policy = ScalingPolicy(
            scale_down_provider_priority=["openrouter", "openai"],
        )
        scaler = AutoScaler(policy=policy)

        metrics = ScalingMetrics(
            agents_by_provider={"openrouter": 2, "openai": 3},
        )

        provider = scaler._select_provider_for_scale_down(metrics)

        assert provider == "openrouter"  # First in priority with agents


class TestAutoScalerApply:
    """Test scaling action application."""

    @pytest.mark.asyncio
    async def test_apply_scale_up(self):
        """Test applying scale-up decision."""
        callback = AsyncMock(return_value=True)
        scaler = AutoScaler(scale_up_callback=callback)

        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_UP,
            reason=ScalingReason.QUEUE_DEPTH,
            recommended_delta=2,
            target_provider="anthropic",
        )

        result = await scaler.apply(decision)

        assert result is True
        callback.assert_called_once_with(decision)
        assert len(scaler._scaling_history) == 1

    @pytest.mark.asyncio
    async def test_apply_scale_down(self):
        """Test applying scale-down decision."""
        callback = AsyncMock(return_value=True)
        scaler = AutoScaler(scale_down_callback=callback)

        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_DOWN,
            reason=ScalingReason.IDLE_AGENTS,
            recommended_delta=-1,
            target_provider="openrouter",
        )

        result = await scaler.apply(decision)

        assert result is True
        callback.assert_called_once_with(decision)

    @pytest.mark.asyncio
    async def test_apply_no_callback(self):
        """Test apply with no callback configured."""
        scaler = AutoScaler()

        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_UP,
            reason=ScalingReason.QUEUE_DEPTH,
            recommended_delta=2,
        )

        result = await scaler.apply(decision)

        assert result is True  # Simulated success

    @pytest.mark.asyncio
    async def test_apply_updates_last_scale_time(self):
        """Test apply updates last scale time."""
        scaler = AutoScaler()
        before = time.time()

        decision = ScalingDecision(
            direction=ScalingDirection.SCALE_UP,
            reason=ScalingReason.QUEUE_DEPTH,
            recommended_delta=2,
        )

        await scaler.apply(decision)

        assert scaler._last_scale_time >= before


class TestAutoScalerSingleton:
    """Test module-level singleton functions."""

    def test_init_auto_scaler(self):
        """Test initializing global auto-scaler."""
        policy = ScalingPolicy(min_agents=5)
        scaler = init_auto_scaler(policy=policy)

        assert get_auto_scaler() is scaler
        assert scaler.policy.min_agents == 5

    def test_set_auto_scaler(self):
        """Test setting global auto-scaler."""
        scaler = AutoScaler()
        set_auto_scaler(scaler)

        assert get_auto_scaler() is scaler


class TestAutoScalerIntegration:
    """Integration tests for auto-scaler."""

    @pytest.mark.asyncio
    async def test_collect_metrics_with_registry(self):
        """Test metrics collection with registry."""
        registry = AsyncMock()
        registry.get_stats = AsyncMock(
            return_value={
                "total_agents": 10,
                "available_agents": 5,
                "busy_agents": 5,
                "by_provider": {"anthropic": 5, "openai": 5},
            }
        )

        scaler = AutoScaler(registry=registry)
        metrics = await scaler._collect_metrics()

        assert metrics.total_agents == 10
        assert metrics.available_agents == 5
        assert metrics.utilization == 0.5

    @pytest.mark.asyncio
    async def test_collect_metrics_with_scheduler(self):
        """Test metrics collection with scheduler."""
        scheduler = AsyncMock()
        scheduler.get_stats = AsyncMock(
            return_value={
                "pending_tasks": 15,
                "running_tasks": 5,
            }
        )

        scaler = AutoScaler(scheduler=scheduler)
        metrics = await scaler._collect_metrics()

        assert metrics.queue_depth == 15
        assert metrics.pending_tasks == 15
        assert metrics.running_tasks == 5

    @pytest.mark.asyncio
    async def test_p99_latency_calculation(self):
        """Test p99 latency calculation from samples."""
        scaler = AutoScaler()

        # Add 100 samples from 1-100ms
        for i in range(1, 101):
            scaler.record_latency(float(i))

        metrics = await scaler._collect_metrics()

        # P99 should be around 99
        assert metrics.latency_p99_ms >= 99
