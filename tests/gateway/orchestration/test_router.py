"""Tests for task router."""

import pytest

from aragora.gateway.external_agents.base import (
    AgentCapability,
    IsolationLevel,
    ExternalAgentTask,
    ExternalAgentResult,
    BaseExternalAgentAdapter,
)
from aragora.gateway.orchestration.router import (
    TaskRouter,
    RoutingStrategy,
    RoutingDecision,
    AgentMetrics,
)


class MockAdapter(BaseExternalAgentAdapter):
    """Mock adapter for testing."""

    def __init__(
        self,
        name: str,
        capabilities: list[AgentCapability],
        isolation_level: IsolationLevel = IsolationLevel.CONTAINER,
    ):
        self._name = name
        self._capabilities = set(capabilities)
        self._isolation_level = isolation_level

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> set[AgentCapability]:
        return self._capabilities

    @property
    def isolation_level(self) -> IsolationLevel:
        return self._isolation_level

    async def execute(self, task: ExternalAgentTask) -> ExternalAgentResult:
        return ExternalAgentResult(
            task_id=task.task_id,
            success=True,
            output=f"Executed by {self._name}",
            agent_name=self._name,
        )


class TestRoutingStrategy:
    """Tests for RoutingStrategy enum."""

    def test_strategy_values(self):
        """Test strategy values."""
        assert RoutingStrategy.CAPABILITY.value == "capability"
        assert RoutingStrategy.COST.value == "cost"
        assert RoutingStrategy.LATENCY.value == "latency"
        assert RoutingStrategy.LOAD_BALANCE.value == "load_balance"
        assert RoutingStrategy.HYBRID.value == "hybrid"


class TestAgentMetrics:
    """Tests for AgentMetrics."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = AgentMetrics(agent_name="test")
        assert metrics.avg_latency_ms == 0.0
        assert metrics.success_rate == 1.0
        assert metrics.current_load == 0
        assert metrics.max_concurrent == 10

    def test_availability_score(self):
        """Test availability score calculation."""
        metrics = AgentMetrics(
            agent_name="test",
            current_load=5,
            max_concurrent=10,
            success_rate=0.9,
        )
        # (1 - 5/10) * 0.9 = 0.5 * 0.9 = 0.45
        assert metrics.availability_score == pytest.approx(0.45)

    def test_availability_score_at_capacity(self):
        """Test availability score when at capacity."""
        metrics = AgentMetrics(
            agent_name="test",
            current_load=10,
            max_concurrent=10,
        )
        assert metrics.availability_score == 0.0

    def test_is_available(self):
        """Test is_available property."""
        metrics = AgentMetrics(
            agent_name="test",
            current_load=5,
            max_concurrent=10,
        )
        assert metrics.is_available is True

        metrics.current_load = 10
        assert metrics.is_available is False


class TestRoutingDecision:
    """Tests for RoutingDecision."""

    def test_decision_creation(self):
        """Test decision creation."""
        decision = RoutingDecision(
            selected_agent="agent-1",
            strategy_used=RoutingStrategy.CAPABILITY,
            score=0.95,
            alternatives=["agent-2", "agent-3"],
            reason="Best match",
        )
        assert decision.selected_agent == "agent-1"
        assert decision.strategy_used == RoutingStrategy.CAPABILITY
        assert decision.score == 0.95
        assert len(decision.alternatives) == 2


class TestTaskRouter:
    """Tests for TaskRouter."""

    def test_register_agent(self):
        """Test registering an agent."""
        router = TaskRouter()
        adapter = MockAdapter("agent-1", [AgentCapability.WEB_SEARCH])

        router.register_agent(
            "agent-1",
            adapter,
            [AgentCapability.WEB_SEARCH],
            priority=1,
            cost_per_request=0.01,
        )

        assert "agent-1" in router._adapters
        assert "agent-1" in router._capabilities
        assert "agent-1" in router._metrics

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        router = TaskRouter()
        adapter = MockAdapter("agent-1", [AgentCapability.WEB_SEARCH])
        router.register_agent("agent-1", adapter, [AgentCapability.WEB_SEARCH])

        result = router.unregister_agent("agent-1")
        assert result is True
        assert "agent-1" not in router._adapters

    def test_get_capable_agents(self):
        """Test getting agents by capability."""
        router = TaskRouter()
        router.register_agent(
            "search-agent",
            MockAdapter("search", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )
        router.register_agent(
            "code-agent",
            MockAdapter("code", [AgentCapability.EXECUTE_CODE]),
            [AgentCapability.EXECUTE_CODE],
        )
        router.register_agent(
            "multi-agent",
            MockAdapter(
                "multi",
                [AgentCapability.WEB_SEARCH, AgentCapability.EXECUTE_CODE],
            ),
            [AgentCapability.WEB_SEARCH, AgentCapability.EXECUTE_CODE],
        )

        # Single capability
        capable = router.get_capable_agents([AgentCapability.WEB_SEARCH])
        assert "search-agent" in capable
        assert "multi-agent" in capable
        assert "code-agent" not in capable

        # Multiple capabilities
        capable = router.get_capable_agents(
            [
                AgentCapability.WEB_SEARCH,
                AgentCapability.EXECUTE_CODE,
            ]
        )
        assert "multi-agent" in capable
        assert "search-agent" not in capable

    @pytest.mark.asyncio
    async def test_route_capability_strategy(self):
        """Test routing with capability strategy."""
        router = TaskRouter(default_strategy=RoutingStrategy.CAPABILITY)
        router.register_agent(
            "agent-1",
            MockAdapter("agent-1", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )

        task = ExternalAgentTask(
            prompt="test",
            required_capabilities=[AgentCapability.WEB_SEARCH],
        )

        decision = await router.route(task)
        assert decision.selected_agent == "agent-1"
        assert decision.strategy_used == RoutingStrategy.CAPABILITY

    @pytest.mark.asyncio
    async def test_route_cost_strategy(self):
        """Test routing with cost optimization."""
        router = TaskRouter(default_strategy=RoutingStrategy.COST)
        router.register_agent(
            "expensive",
            MockAdapter("expensive", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
            cost_per_request=1.0,
        )
        router.register_agent(
            "cheap",
            MockAdapter("cheap", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
            cost_per_request=0.01,
        )

        task = ExternalAgentTask(
            prompt="test",
            required_capabilities=[AgentCapability.WEB_SEARCH],
        )

        decision = await router.route(task, strategy=RoutingStrategy.COST)
        assert decision.selected_agent == "cheap"

    @pytest.mark.asyncio
    async def test_route_latency_strategy(self):
        """Test routing with latency optimization."""
        router = TaskRouter(default_strategy=RoutingStrategy.LATENCY)
        router.register_agent(
            "slow",
            MockAdapter("slow", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )
        router.register_agent(
            "fast",
            MockAdapter("fast", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )
        # Set latency metrics
        router._metrics["slow"].avg_latency_ms = 1000
        router._metrics["fast"].avg_latency_ms = 100

        task = ExternalAgentTask(
            prompt="test",
            required_capabilities=[AgentCapability.WEB_SEARCH],
        )

        decision = await router.route(task, strategy=RoutingStrategy.LATENCY)
        assert decision.selected_agent == "fast"

    @pytest.mark.asyncio
    async def test_route_load_balance_strategy(self):
        """Test routing with load balancing."""
        router = TaskRouter(default_strategy=RoutingStrategy.LOAD_BALANCE)
        router.register_agent(
            "busy",
            MockAdapter("busy", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )
        router.register_agent(
            "idle",
            MockAdapter("idle", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )
        # Set load metrics
        router._metrics["busy"].current_load = 9
        router._metrics["idle"].current_load = 1

        task = ExternalAgentTask(
            prompt="test",
            required_capabilities=[AgentCapability.WEB_SEARCH],
        )

        decision = await router.route(task, strategy=RoutingStrategy.LOAD_BALANCE)
        assert decision.selected_agent == "idle"

    @pytest.mark.asyncio
    async def test_route_no_capable_agents(self):
        """Test routing when no agents have required capabilities."""
        router = TaskRouter()
        router.register_agent(
            "agent-1",
            MockAdapter("agent-1", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )

        task = ExternalAgentTask(
            prompt="test",
            required_capabilities=[AgentCapability.SHELL_ACCESS],  # Not available
        )

        with pytest.raises(ValueError):
            await router.route(task)

    @pytest.mark.asyncio
    async def test_route_with_exclusions(self):
        """Test routing with excluded agents."""
        router = TaskRouter()
        router.register_agent(
            "agent-1",
            MockAdapter("agent-1", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )
        router.register_agent(
            "agent-2",
            MockAdapter("agent-2", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )

        task = ExternalAgentTask(
            prompt="test",
            required_capabilities=[AgentCapability.WEB_SEARCH],
        )

        decision = await router.route(task, exclude_agents=["agent-1"])
        assert decision.selected_agent == "agent-2"

    def test_update_metrics(self):
        """Test updating agent metrics after execution."""
        router = TaskRouter()
        router.register_agent(
            "agent-1",
            MockAdapter("agent-1", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )

        # Update with success
        router.update_metrics("agent-1", latency_ms=100.0, success=True)
        assert router._metrics["agent-1"].avg_latency_ms > 0

        # Update with failure
        router.update_metrics("agent-1", latency_ms=500.0, success=False)
        assert router._metrics["agent-1"].error_count_24h > 0

    def test_increment_decrement_load(self):
        """Test load tracking."""
        router = TaskRouter()
        router.register_agent(
            "agent-1",
            MockAdapter("agent-1", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
        )

        assert router._metrics["agent-1"].current_load == 0

        router.increment_load("agent-1")
        assert router._metrics["agent-1"].current_load == 1

        router.decrement_load("agent-1")
        assert router._metrics["agent-1"].current_load == 0

        # Don't go negative
        router.decrement_load("agent-1")
        assert router._metrics["agent-1"].current_load == 0

    def test_get_agent_status(self):
        """Test getting agent status."""
        router = TaskRouter()
        router.register_agent(
            "agent-1",
            MockAdapter("agent-1", [AgentCapability.WEB_SEARCH]),
            [AgentCapability.WEB_SEARCH],
            priority=5,
        )

        status = router.get_agent_status()
        assert len(status) == 1
        assert status[0]["name"] == "agent-1"
        assert status[0]["priority"] == 5
        assert "metrics" in status[0]
