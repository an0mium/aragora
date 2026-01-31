"""Tests for fallback chain."""

import pytest
import asyncio

from aragora.gateway.external_agents.base import (
    AgentCapability,
    IsolationLevel,
    ExternalAgentTask,
    ExternalAgentResult,
    BaseExternalAgentAdapter,
)
from aragora.gateway.orchestration.fallback import (
    FallbackChain,
    FallbackResult,
    FallbackReason,
    CircuitState,
)


class MockAdapter(BaseExternalAgentAdapter):
    """Mock adapter for testing."""

    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        fail_count: int = 0,
        error_message: str = "Mock failure",
    ):
        self._name = name
        self._should_fail = should_fail
        self._fail_count = fail_count
        self._error_message = error_message
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> set[AgentCapability]:
        return {AgentCapability.WEB_SEARCH}

    @property
    def isolation_level(self) -> IsolationLevel:
        return IsolationLevel.CONTAINER

    async def execute(self, task: ExternalAgentTask) -> ExternalAgentResult:
        self._call_count += 1

        if self._should_fail:
            return ExternalAgentResult(
                task_id=task.task_id,
                success=False,
                error=self._error_message,
                agent_name=self._name,
            )

        if self._fail_count > 0 and self._call_count <= self._fail_count:
            return ExternalAgentResult(
                task_id=task.task_id,
                success=False,
                error=f"Temporary failure {self._call_count}/{self._fail_count}",
                agent_name=self._name,
            )

        return ExternalAgentResult(
            task_id=task.task_id,
            success=True,
            output=f"Success from {self._name}",
            agent_name=self._name,
        )


class TestFallbackReason:
    """Tests for FallbackReason enum."""

    def test_reason_values(self):
        """Test reason values."""
        assert FallbackReason.TIMEOUT.value == "timeout"
        assert FallbackReason.ERROR.value == "error"
        assert FallbackReason.RATE_LIMITED.value == "rate_limited"
        assert FallbackReason.CIRCUIT_OPEN.value == "circuit_open"


class TestCircuitState:
    """Tests for CircuitState."""

    def test_default_state(self):
        """Test default circuit state."""
        state = CircuitState(agent_name="test")
        assert state.state == "closed"
        assert state.failure_count == 0
        assert state.is_open is False

    def test_record_failure_opens_circuit(self):
        """Test that failures open the circuit."""
        state = CircuitState(
            agent_name="test",
            opens_at_failures=3,
        )

        for _ in range(3):
            state.record_failure()

        assert state.state == "open"
        assert state.is_open is True

    def test_record_success_closes_half_open(self):
        """Test that success closes half-open circuit."""
        state = CircuitState(
            agent_name="test",
            opens_at_failures=1,
        )
        state.record_failure()
        assert state.state == "open"

        # Manually set to half-open (simulating timeout)
        state.state = "half-open"

        state.record_success()
        assert state.state == "closed"
        assert state.failure_count == 0


class TestFallbackResult:
    """Tests for FallbackResult."""

    def test_success_result(self):
        """Test successful result."""
        result = FallbackResult(
            success=True,
            final_agent="agent-1",
            attempts=1,
        )
        assert result.success is True
        assert result.used_fallback is False

    def test_fallback_result(self):
        """Test result with fallback."""
        result = FallbackResult(
            success=True,
            final_agent="agent-2",
            attempts=2,
            fallback_chain=["agent-1", "agent-2"],
            fallback_reasons={"agent-1": FallbackReason.ERROR},
            degraded=True,
        )
        assert result.success is True
        assert result.used_fallback is True
        assert result.degraded is True


class TestFallbackChain:
    """Tests for FallbackChain."""

    def test_add_agent(self):
        """Test adding agents to chain."""
        chain = FallbackChain()
        adapter = MockAdapter("agent-1")

        chain.add_agent("agent-1", adapter, priority=1)

        assert "agent-1" in chain._adapters
        assert "agent-1" in chain._priorities
        assert "agent-1" in chain._circuits

    def test_remove_agent(self):
        """Test removing agent from chain."""
        chain = FallbackChain()
        adapter = MockAdapter("agent-1")
        chain.add_agent("agent-1", adapter)

        result = chain.remove_agent("agent-1")
        assert result is True
        assert "agent-1" not in chain._adapters

    @pytest.mark.asyncio
    async def test_execute_success_first_agent(self):
        """Test successful execution on first agent."""
        chain = FallbackChain()
        chain.add_agent("agent-1", MockAdapter("agent-1"), priority=1)
        chain.add_agent("agent-2", MockAdapter("agent-2"), priority=2)

        task = ExternalAgentTask(prompt="test")
        result = await chain.execute(task)

        assert result.success is True
        assert result.final_agent == "agent-1"
        assert result.attempts == 1
        assert result.used_fallback is False

    @pytest.mark.asyncio
    async def test_execute_fallback_on_failure(self):
        """Test fallback to second agent on first failure."""
        chain = FallbackChain(max_retries=1)  # Fail fast for test
        chain.add_agent(
            "failing-agent",
            MockAdapter("failing", should_fail=True),
            priority=1,
        )
        chain.add_agent(
            "success-agent",
            MockAdapter("success"),
            priority=2,
        )

        task = ExternalAgentTask(prompt="test")
        result = await chain.execute(task)

        assert result.success is True
        assert result.final_agent == "success-agent"
        assert result.attempts == 2
        assert result.used_fallback is True
        assert "failing-agent" in result.fallback_reasons

    @pytest.mark.asyncio
    async def test_execute_all_agents_fail(self):
        """Test when all agents fail."""
        chain = FallbackChain(max_retries=1)
        chain.add_agent(
            "agent-1",
            MockAdapter("agent-1", should_fail=True),
            priority=1,
        )
        chain.add_agent(
            "agent-2",
            MockAdapter("agent-2", should_fail=True),
            priority=2,
        )

        task = ExternalAgentTask(prompt="test")
        result = await chain.execute(task)

        assert result.success is False
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_execute_respects_priority(self):
        """Test that execution respects agent priority."""
        chain = FallbackChain()
        chain.add_agent("low-priority", MockAdapter("low"), priority=10)
        chain.add_agent("high-priority", MockAdapter("high"), priority=1)

        task = ExternalAgentTask(prompt="test")
        result = await chain.execute(task)

        assert result.final_agent == "high-priority"

    @pytest.mark.asyncio
    async def test_execute_with_skip_agents(self):
        """Test execution with skipped agents."""
        chain = FallbackChain()
        chain.add_agent("agent-1", MockAdapter("agent-1"), priority=1)
        chain.add_agent("agent-2", MockAdapter("agent-2"), priority=2)

        task = ExternalAgentTask(prompt="test")
        result = await chain.execute(task, skip_agents=["agent-1"])

        assert result.final_agent == "agent-2"

    @pytest.mark.asyncio
    async def test_execute_with_start_from(self):
        """Test execution starting from specific agent."""
        chain = FallbackChain()
        chain.add_agent("agent-1", MockAdapter("agent-1"), priority=1)
        chain.add_agent("agent-2", MockAdapter("agent-2"), priority=2)
        chain.add_agent("agent-3", MockAdapter("agent-3"), priority=3)

        task = ExternalAgentTask(prompt="test")
        result = await chain.execute(task, start_from="agent-2")

        assert result.final_agent == "agent-2"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after failures."""
        chain = FallbackChain(
            max_retries=1,
            enable_circuit_breaker=True,
        )
        chain.add_agent(
            "failing-agent",
            MockAdapter("failing", should_fail=True),
            priority=1,
            circuit_threshold=2,
        )
        chain.add_agent(
            "success-agent",
            MockAdapter("success"),
            priority=2,
        )

        task = ExternalAgentTask(prompt="test")

        # First execution - fails and increments circuit counter
        await chain.execute(task)

        # Second execution - fails and opens circuit
        await chain.execute(task)

        # Circuit should be open now
        status = chain.get_circuit_status()
        assert status["failing-agent"]["state"] == "open"
        assert status["failing-agent"]["is_open"] is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_skips_open_circuits(self):
        """Test that open circuits are skipped."""
        chain = FallbackChain(
            max_retries=1,
            enable_circuit_breaker=True,
        )
        chain.add_agent(
            "open-circuit",
            MockAdapter("open", should_fail=True),
            priority=1,
            circuit_threshold=1,
        )
        chain.add_agent(
            "success-agent",
            MockAdapter("success"),
            priority=2,
        )

        task = ExternalAgentTask(prompt="test")

        # First execution - opens the circuit
        await chain.execute(task)

        # Second execution - should skip to success agent
        result = await chain.execute(task)
        assert result.final_agent == "success-agent"
        assert result.attempts == 1  # Only tried success agent

    def test_reset_circuit(self):
        """Test manually resetting a circuit."""
        chain = FallbackChain()
        chain.add_agent(
            "agent-1",
            MockAdapter("agent-1"),
            priority=1,
        )

        # Force circuit open
        chain._circuits["agent-1"].state = "open"
        chain._circuits["agent-1"].failure_count = 5

        result = chain.reset_circuit("agent-1")
        assert result is True
        assert chain._circuits["agent-1"].state == "closed"
        assert chain._circuits["agent-1"].failure_count == 0

    def test_reset_all_circuits(self):
        """Test resetting all circuits."""
        chain = FallbackChain()
        chain.add_agent("agent-1", MockAdapter("agent-1"), priority=1)
        chain.add_agent("agent-2", MockAdapter("agent-2"), priority=2)

        # Force circuits open
        chain._circuits["agent-1"].state = "open"
        chain._circuits["agent-2"].state = "open"

        chain.reset_all_circuits()

        assert chain._circuits["agent-1"].state == "closed"
        assert chain._circuits["agent-2"].state == "closed"

    @pytest.mark.asyncio
    async def test_fallback_callback(self):
        """Test fallback callback is called."""
        chain = FallbackChain(max_retries=1)
        fallback_events = []

        async def on_fallback(from_agent, to_agent, reason):
            fallback_events.append((from_agent, to_agent, reason))

        chain.on_fallback(on_fallback)
        chain.add_agent(
            "failing",
            MockAdapter("failing", should_fail=True),
            priority=1,
        )
        chain.add_agent(
            "success",
            MockAdapter("success"),
            priority=2,
        )

        task = ExternalAgentTask(prompt="test")
        await chain.execute(task)

        assert len(fallback_events) == 1
        assert fallback_events[0][0] == "failing"
        assert fallback_events[0][1] == "success"

    @pytest.mark.asyncio
    async def test_exhausted_callback(self):
        """Test exhausted callback is called when all agents fail."""
        chain = FallbackChain(max_retries=1)
        exhausted_tasks = []

        async def on_exhausted(task):
            exhausted_tasks.append(task)

        chain.on_exhausted(on_exhausted)
        chain.add_agent(
            "failing",
            MockAdapter("failing", should_fail=True),
            priority=1,
        )

        task = ExternalAgentTask(prompt="test")
        await chain.execute(task)

        assert len(exhausted_tasks) == 1
        assert exhausted_tasks[0] == task
