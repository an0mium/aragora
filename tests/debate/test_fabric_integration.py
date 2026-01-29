"""Tests for Arena-Fabric integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.fabric.fabric import AgentFabric, AgentPool
from aragora.fabric.models import (
    AgentConfig,
    AgentHandle,
    BudgetConfig,
    BudgetStatus,
    HealthStatus,
    Policy,
    PolicyContext,
    PolicyDecision,
    PolicyEffect,
    PolicyRule,
    Priority,
    Task,
    TaskHandle,
    TaskStatus,
    Usage,
)
from aragora.debate.fabric_integration import (
    FabricAgentAdapter,
    FabricDebateConfig,
    FabricDebateRunner,
    FabricUsageTracker,
    create_debate_policy,
    register_debate_executor,
)


@pytest.fixture
def fabric():
    return AgentFabric()


@pytest.fixture
def mock_fabric():
    """Create a mock AgentFabric for testing."""
    fabric = MagicMock(spec=AgentFabric)

    # Mock pool
    pool = AgentPool(
        id="pool-test",
        name="test-pool",
        model="claude-3-opus",
        current_agents=["agent-1", "agent-2", "agent-3"],
    )

    # Configure async methods
    fabric.get_pool = AsyncMock(return_value=pool)
    fabric.check_policy = AsyncMock(
        return_value=PolicyDecision(allowed=True, effect=PolicyEffect.ALLOW)
    )
    fabric.schedule = AsyncMock(
        return_value=TaskHandle(
            task_id="task-1",
            agent_id="agent-1",
            status=TaskStatus.SCHEDULED,
            scheduled_at=MagicMock(),
        )
    )
    fabric.complete_task = AsyncMock()
    fabric.track_usage = AsyncMock(
        return_value=BudgetStatus(
            entity_id="pool-test",
            entity_type="agent",
            period_start=MagicMock(),
            period_end=MagicMock(),
        )
    )
    fabric.check_budget = AsyncMock(
        return_value=(
            True,
            BudgetStatus(
                entity_id="pool-test",
                entity_type="agent",
                period_start=MagicMock(),
                period_end=MagicMock(),
            ),
        )
    )
    fabric.get_agent = AsyncMock(
        return_value=AgentHandle(
            agent_id="agent-1",
            config=AgentConfig(id="agent-1", model="claude-3-opus"),
            spawned_at=MagicMock(),
        )
    )

    return fabric


class TestFabricDebateConfig:
    def test_default_config(self):
        config = FabricDebateConfig(pool_id="pool-1")
        assert config.pool_id == "pool-1"
        assert config.priority == Priority.NORMAL
        assert config.timeout_seconds == 600.0
        assert config.min_agents == 2
        assert config.max_agents == 10
        assert config.require_policy_check is True

    def test_custom_config(self):
        config = FabricDebateConfig(
            pool_id="pool-1",
            budget_per_debate_usd=5.0,
            priority=Priority.HIGH,
            min_agents=3,
            org_id="org-1",
            user_id="user-1",
        )
        assert config.budget_per_debate_usd == 5.0
        assert config.priority == Priority.HIGH
        assert config.min_agents == 3
        assert config.org_id == "org-1"


class TestFabricUsageTracker:
    @pytest.mark.asyncio
    async def test_track_usage(self, mock_fabric):
        tracker = FabricUsageTracker(
            fabric=mock_fabric,
            entity_id="pool-1",
            debate_id="debate-1",
        )

        result = await tracker.track(
            tokens_input=100,
            tokens_output=50,
            cost_usd=0.01,
            agent_id="agent-1",
            model="claude-3-opus",
        )

        assert result is True
        assert tracker.total_cost == 0.01
        assert tracker.total_tokens == 150
        assert "agent-1" in tracker.per_agent_cost

        mock_fabric.track_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_multiple_agents(self, mock_fabric):
        tracker = FabricUsageTracker(
            fabric=mock_fabric,
            entity_id="pool-1",
            debate_id="debate-1",
        )

        await tracker.track(100, 50, 0.01, "agent-1")
        await tracker.track(200, 100, 0.02, "agent-2")
        await tracker.track(150, 75, 0.015, "agent-1")

        assert tracker.total_cost == 0.045
        assert tracker.total_tokens == 675
        assert tracker.per_agent_cost["agent-1"] == 0.025
        assert tracker.per_agent_cost["agent-2"] == 0.02

    @pytest.mark.asyncio
    async def test_budget_exceeded(self, mock_fabric):
        tracker = FabricUsageTracker(
            fabric=mock_fabric,
            entity_id="pool-1",
            debate_id="debate-1",
            budget_limit_usd=0.05,
        )

        # First call under budget
        result = await tracker.track(100, 50, 0.03, "agent-1")
        assert result is True

        # Second call exceeds budget
        result = await tracker.track(100, 50, 0.03, "agent-1")
        assert result is False


class TestFabricAgentAdapter:
    @pytest.mark.asyncio
    async def test_generate(self, mock_fabric):
        adapter = FabricAgentAdapter(
            fabric=mock_fabric,
            agent_id="agent-1",
            model="claude-3-opus",
        )

        assert adapter.name == "agent-1"
        assert adapter.model == "claude-3-opus"

        result = await adapter.generate("Test prompt")

        assert "[Fabric-managed response" in result
        mock_fabric.check_policy.assert_called_once()
        mock_fabric.check_budget.assert_called_once()
        mock_fabric.schedule.assert_called_once()
        mock_fabric.complete_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_policy_denied(self, mock_fabric):
        mock_fabric.check_policy = AsyncMock(
            return_value=PolicyDecision(
                allowed=False,
                effect=PolicyEffect.DENY,
                reason="Test denial",
            )
        )

        adapter = FabricAgentAdapter(
            fabric=mock_fabric,
            agent_id="agent-1",
            model="claude-3-opus",
        )

        with pytest.raises(PermissionError, match="Policy denied"):
            await adapter.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_budget_exceeded(self, mock_fabric):
        mock_fabric.check_budget = AsyncMock(
            return_value=(
                False,
                BudgetStatus(
                    entity_id="agent-1",
                    entity_type="agent",
                    period_start=MagicMock(),
                    period_end=MagicMock(),
                    over_limit=True,
                ),
            )
        )

        adapter = FabricAgentAdapter(
            fabric=mock_fabric,
            agent_id="agent-1",
            model="claude-3-opus",
        )

        with pytest.raises(RuntimeError, match="Budget exceeded"):
            await adapter.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_with_usage_tracker(self, mock_fabric):
        tracker = FabricUsageTracker(
            fabric=mock_fabric,
            entity_id="pool-1",
            debate_id="debate-1",
        )

        adapter = FabricAgentAdapter(
            fabric=mock_fabric,
            agent_id="agent-1",
            model="claude-3-opus",
            usage_tracker=tracker,
        )

        await adapter.generate("Test prompt")

        # Usage tracker should have recorded usage
        assert tracker.total_tokens > 0


class TestFabricDebateRunner:
    @pytest.mark.asyncio
    async def test_run_debate_pool_not_found(self, mock_fabric):
        mock_fabric.get_pool = AsyncMock(return_value=None)

        runner = FabricDebateRunner(mock_fabric)

        with pytest.raises(ValueError, match="not found"):
            from aragora.core_types import Environment

            await runner.run_debate(
                environment=Environment(task="Test task"),
                pool_id="nonexistent-pool",
            )

    @pytest.mark.asyncio
    async def test_run_debate_insufficient_agents(self, mock_fabric):
        mock_fabric.get_pool = AsyncMock(
            return_value=AgentPool(
                id="pool-test",
                name="test-pool",
                model="claude-3-opus",
                current_agents=["agent-1"],  # Only 1 agent
            )
        )

        runner = FabricDebateRunner(mock_fabric)

        with pytest.raises(RuntimeError, match="minimum required"):
            from aragora.core_types import Environment

            await runner.run_debate(
                environment=Environment(task="Test task"),
                pool_id="pool-test",
                config=FabricDebateConfig(pool_id="pool-test", min_agents=3),
            )

    @pytest.mark.asyncio
    async def test_run_debate_policy_denied(self, mock_fabric):
        mock_fabric.check_policy = AsyncMock(
            return_value=PolicyDecision(
                allowed=False,
                effect=PolicyEffect.DENY,
                reason="Debates not allowed",
            )
        )

        runner = FabricDebateRunner(mock_fabric)

        with pytest.raises(PermissionError, match="Policy denied"):
            from aragora.core_types import Environment

            await runner.run_debate(
                environment=Environment(task="Test task"),
                pool_id="pool-test",
            )

    @pytest.mark.asyncio
    async def test_get_active_debates(self, mock_fabric):
        runner = FabricDebateRunner(mock_fabric)

        # Initially empty
        active = await runner.get_active_debates()
        assert active == []

    @pytest.mark.asyncio
    async def test_cancel_debate_not_found(self, mock_fabric):
        runner = FabricDebateRunner(mock_fabric)

        result = await runner.cancel_debate("nonexistent-debate")
        assert result is False


class TestCreateDebatePolicy:
    def test_default_policy(self):
        policy = create_debate_policy()

        assert policy.name == "default-debate-policy"
        assert len(policy.rules) == 2
        assert policy.priority == 10
        assert policy.enabled is True

    def test_custom_policy(self):
        policy = create_debate_policy(
            name="custom-policy",
            max_agents=5,
            max_cost_per_debate=1.0,
            allowed_models=["claude-3-opus", "gpt-4"],
        )

        assert policy.name == "custom-policy"
        assert len(policy.rules) == 3  # 2 default + 1 model restriction

    def test_policy_metadata(self):
        policy = create_debate_policy(max_cost_per_debate=5.0)

        assert policy.metadata["max_cost_per_debate"] == "5.0"
        assert policy.metadata["type"] == "debate"


class TestRegisterDebateExecutor:
    @pytest.mark.asyncio
    async def test_register(self, mock_fabric):
        mock_fabric.register_executor = MagicMock()

        await register_debate_executor(mock_fabric)

        mock_fabric.register_executor.assert_called_once()
        call_args = mock_fabric.register_executor.call_args
        assert call_args[0][0] == "debate"
        assert callable(call_args[0][1])


class TestArenaFabricIntegration:
    """Integration tests for Arena with fabric."""

    @pytest.mark.asyncio
    async def test_arena_with_fabric_config_validation(self):
        """Test that Arena validates fabric config correctly."""
        from aragora.debate.orchestrator import Arena
        from aragora.core_types import Environment

        # Should fail when specifying both agents and fabric
        env = Environment(task="Test task")

        with pytest.raises(ValueError, match="Cannot specify both"):
            Arena(
                environment=env,
                agents=[MagicMock(name="test-agent")],
                fabric=MagicMock(),
                fabric_config=FabricDebateConfig(pool_id="test"),
            )

    @pytest.mark.asyncio
    async def test_arena_without_agents_or_fabric(self):
        """Test that Arena requires either agents or fabric."""
        from aragora.debate.orchestrator import Arena
        from aragora.core_types import Environment

        env = Environment(task="Test task")

        with pytest.raises(ValueError, match="Must specify either"):
            Arena(environment=env, agents=None)
