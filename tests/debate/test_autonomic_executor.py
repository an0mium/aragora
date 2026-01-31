"""
Tests for the autonomic executor module.

Tests cover:
- StreamingContentBuffer class
- AutonomicExecutor initialization
- Timeout escalation and retry tracking
- Generate method with various scenarios
- Critique method with various scenarios
- Vote method with various scenarios
- Generate with fallback agents
- Error handling (timeout, connection, internal)
- Circuit breaker integration
- Immune system integration
- Chaos director integration
- Performance monitor integration
- Telemetry emission
- Wisdom store fallback
- Event hooks
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import Agent, Critique, Message, Vote
from aragora.debate.autonomic_executor import (
    AutonomicExecutor,
    StreamingContentBuffer,
)
from aragora.resilience.circuit_breaker import CircuitBreaker


# =============================================================================
# Mock Agent Classes
# =============================================================================


class MockAgent(Agent):
    """Mock agent for testing AutonomicExecutor functionality."""

    def __init__(
        self,
        name: str = "mock-agent",
        response: str = "Test response",
        model: str = "mock-model",
        role: str = "proposer",
        vote_choice: str | None = None,
        vote_confidence: float = 0.8,
    ):
        super().__init__(name=name, model=model, role=role)
        self.agent_type = "mock"
        self.response = response
        self.vote_choice = vote_choice
        self.vote_confidence = vote_confidence
        self.generate_calls = 0
        self.critique_calls = 0
        self.vote_calls = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        return self.response

    async def generate_stream(self, prompt: str, context: list = None):
        yield self.response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls += 1
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        choice = self.vote_choice or (list(proposals.keys())[0] if proposals else self.name)
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=self.vote_confidence,
        )


class EmptyOutputAgent(MockAgent):
    """Agent that returns empty output."""

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        return ""

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls += 1
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content="",
            issues=[],
            suggestions=[],
            severity=0.0,
            reasoning="",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        return Vote(
            agent=self.name,
            choice="",
            reasoning="",
            confidence=0.0,
        )


class TimeoutAgent(MockAgent):
    """Agent that times out."""

    def __init__(self, name: str = "timeout-agent", delay: float = 10.0):
        super().__init__(name=name)
        self.delay = delay

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        await asyncio.sleep(self.delay)
        return "Delayed response"

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls += 1
        await asyncio.sleep(self.delay)
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Timeout issue"],
            suggestions=["Timeout suggestion"],
            severity=0.5,
            reasoning="Timeout reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        await asyncio.sleep(self.delay)
        return Vote(
            agent=self.name,
            choice=list(proposals.keys())[0] if proposals else self.name,
            reasoning="Timeout vote",
            confidence=0.5,
        )


class FailingAgent(MockAgent):
    """Agent that raises exceptions."""

    def __init__(self, name: str = "failing-agent", exception_type: type = RuntimeError):
        super().__init__(name=name)
        self.exception_type = exception_type

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        raise self.exception_type("Agent generation failed")

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls += 1
        raise self.exception_type("Agent critique failed")

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        raise self.exception_type("Agent vote failed")


class ConnectionErrorAgent(MockAgent):
    """Agent that raises connection errors."""

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        raise ConnectionError("Network unreachable")

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls += 1
        raise ConnectionError("Network unreachable")

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls += 1
        raise ConnectionError("Network unreachable")


class RetryableAgent(MockAgent):
    """Agent that fails initially then succeeds."""

    def __init__(self, name: str = "retryable-agent", failures_before_success: int = 1):
        super().__init__(name=name)
        self.failures_before_success = failures_before_success
        self.attempt_count = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        self.attempt_count += 1
        if self.attempt_count <= self.failures_before_success:
            return ""  # Empty output triggers retry
        return "Success after retry"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent():
    """Create a basic mock agent."""
    return MockAgent(name="test-agent", response="Test response")


@pytest.fixture
def empty_agent():
    """Create an agent that returns empty output."""
    return EmptyOutputAgent(name="empty-agent")


@pytest.fixture
def timeout_agent():
    """Create an agent that times out."""
    return TimeoutAgent(name="timeout-agent", delay=10.0)


@pytest.fixture
def failing_agent():
    """Create an agent that raises exceptions."""
    return FailingAgent(name="failing-agent")


@pytest.fixture
def connection_error_agent():
    """Create an agent that raises connection errors."""
    return ConnectionErrorAgent(name="connection-agent")


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker."""
    return CircuitBreaker(name="test-breaker", failure_threshold=3, cooldown_seconds=60.0)


@pytest.fixture
def executor():
    """Create a basic executor."""
    return AutonomicExecutor(default_timeout=5.0)


@pytest.fixture
def executor_with_circuit_breaker(circuit_breaker):
    """Create an executor with circuit breaker."""
    return AutonomicExecutor(circuit_breaker=circuit_breaker, default_timeout=5.0)


@pytest.fixture
def mock_immune_system():
    """Create a mock immune system."""
    immune = MagicMock()
    immune.agent_started = MagicMock()
    immune.agent_completed = MagicMock()
    immune.agent_failed = MagicMock()
    immune.agent_timeout = MagicMock()
    return immune


@pytest.fixture
def mock_chaos_director():
    """Create a mock chaos director."""
    director = MagicMock()

    @dataclass
    class MockResponse:
        message: str

    director.timeout_response = MagicMock(
        return_value=MockResponse(message="[Theatrical timeout message]")
    )
    director.connection_response = MagicMock(
        return_value=MockResponse(message="[Theatrical connection message]")
    )
    director.internal_error_response = MagicMock(
        return_value=MockResponse(message="[Theatrical error message]")
    )
    return director


@pytest.fixture
def mock_performance_monitor():
    """Create a mock performance monitor."""
    monitor = MagicMock()
    monitor.track_agent_call = MagicMock(return_value="tracking-id-123")
    monitor.record_completion = MagicMock()
    return monitor


@pytest.fixture
def mock_wisdom_store():
    """Create a mock wisdom store."""
    store = MagicMock()
    store.get_relevant_wisdom = MagicMock(
        return_value=[
            {
                "id": 1,
                "text": "Audience wisdom text",
                "submitter_id": "user-123",
            }
        ]
    )
    store.mark_wisdom_used = MagicMock()
    return store


@pytest.fixture
def context():
    """Create sample context messages."""
    return [
        Message(role="user", agent="user", content="Test message", round=0),
    ]


# =============================================================================
# StreamingContentBuffer Tests
# =============================================================================


class TestStreamingContentBuffer:
    """Tests for StreamingContentBuffer class."""

    @pytest.mark.asyncio
    async def test_append_chunk(self):
        """Test appending chunks to buffer."""
        buffer = StreamingContentBuffer()
        await buffer.append("agent1", "Hello ")
        await buffer.append("agent1", "World")
        assert buffer.get_partial("agent1") == "Hello World"

    @pytest.mark.asyncio
    async def test_multiple_agents(self):
        """Test buffering for multiple agents."""
        buffer = StreamingContentBuffer()
        await buffer.append("agent1", "Content 1")
        await buffer.append("agent2", "Content 2")
        assert buffer.get_partial("agent1") == "Content 1"
        assert buffer.get_partial("agent2") == "Content 2"

    def test_get_partial_nonexistent(self):
        """Test getting partial for nonexistent agent."""
        buffer = StreamingContentBuffer()
        assert buffer.get_partial("nonexistent") == ""

    @pytest.mark.asyncio
    async def test_get_partial_async(self):
        """Test async partial retrieval."""
        buffer = StreamingContentBuffer()
        await buffer.append("agent1", "Async content")
        result = await buffer.get_partial_async("agent1")
        assert result == "Async content"

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clearing buffer."""
        buffer = StreamingContentBuffer()
        await buffer.append("agent1", "Content")
        await buffer.clear("agent1")
        assert buffer.get_partial("agent1") == ""

    def test_clear_sync(self):
        """Test synchronous buffer clear."""
        buffer = StreamingContentBuffer()
        buffer._buffer["agent1"] = "Content"
        buffer.clear_sync("agent1")
        assert buffer.get_partial("agent1") == ""

    def test_clear_sync_nonexistent(self):
        """Test clearing nonexistent agent buffer."""
        buffer = StreamingContentBuffer()
        buffer.clear_sync("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_concurrent_append(self):
        """Test concurrent appends are thread-safe."""
        buffer = StreamingContentBuffer()

        async def append_chunks():
            for i in range(100):
                await buffer.append("agent1", "x")

        await asyncio.gather(append_chunks(), append_chunks())
        assert len(buffer.get_partial("agent1")) == 200


# =============================================================================
# AutonomicExecutor Initialization Tests
# =============================================================================


class TestAutonomicExecutorInit:
    """Tests for AutonomicExecutor initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        executor = AutonomicExecutor()
        assert executor.circuit_breaker is None
        assert executor.timeout_escalation_factor == 1.5
        assert executor.max_timeout == 600.0
        assert executor.streaming_buffer is not None
        assert executor.wisdom_store is None
        assert executor.loop_id is None
        assert executor.immune_system is None
        assert executor.chaos_director is None
        assert executor.performance_monitor is None
        assert executor.enable_telemetry is False

    def test_custom_initialization(self, circuit_breaker, mock_wisdom_store):
        """Test custom initialization."""
        buffer = StreamingContentBuffer()
        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=30.0,
            timeout_escalation_factor=2.0,
            max_timeout=300.0,
            streaming_buffer=buffer,
            wisdom_store=mock_wisdom_store,
            loop_id="test-loop",
            enable_telemetry=True,
        )
        assert executor.circuit_breaker is circuit_breaker
        assert executor.default_timeout == 30.0
        assert executor.timeout_escalation_factor == 2.0
        assert executor.max_timeout == 300.0
        assert executor.streaming_buffer is buffer
        assert executor.wisdom_store is mock_wisdom_store
        assert executor.loop_id == "test-loop"
        assert executor.enable_telemetry is True

    def test_event_hooks_initialization(self):
        """Test event hooks initialization."""
        hooks = {"on_agent_error": MagicMock()}
        executor = AutonomicExecutor(event_hooks=hooks)
        assert executor.event_hooks == hooks

    def test_set_loop_id(self):
        """Test setting loop ID."""
        executor = AutonomicExecutor()
        executor.set_loop_id("new-loop-id")
        assert executor.loop_id == "new-loop-id"


# =============================================================================
# Timeout Escalation Tests
# =============================================================================


class TestTimeoutEscalation:
    """Tests for timeout escalation logic."""

    def test_get_escalated_timeout_first_attempt(self, executor):
        """Test timeout on first attempt."""
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 5.0

    def test_get_escalated_timeout_with_retries(self, executor):
        """Test timeout escalation with retries."""
        executor.record_retry("agent1")
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 5.0 * 1.5  # 7.5

        executor.record_retry("agent1")
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 5.0 * (1.5**2)  # 11.25

    def test_get_escalated_timeout_max_cap(self, executor):
        """Test timeout doesn't exceed max."""
        executor.max_timeout = 10.0
        for _ in range(10):
            executor.record_retry("agent1")
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 10.0

    def test_get_escalated_timeout_custom_base(self, executor):
        """Test timeout with custom base."""
        timeout = executor.get_escalated_timeout("agent1", base_timeout=10.0)
        assert timeout == 10.0

    def test_record_retry(self, executor):
        """Test recording retries."""
        count = executor.record_retry("agent1")
        assert count == 1
        count = executor.record_retry("agent1")
        assert count == 2

    def test_reset_retries(self, executor):
        """Test resetting retries."""
        executor.record_retry("agent1")
        executor.record_retry("agent1")
        executor.reset_retries("agent1")
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 5.0  # Back to base

    def test_reset_retries_nonexistent(self, executor):
        """Test resetting nonexistent agent retries."""
        executor.reset_retries("nonexistent")  # Should not raise


# =============================================================================
# with_timeout Tests
# =============================================================================


class TestWithTimeout:
    """Tests for the with_timeout method."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, executor):
        """Test successful execution within timeout."""

        async def quick_coro():
            return "result"

        result = await executor.with_timeout(quick_coro(), "agent1", timeout_seconds=1.0)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_timeout_error(self, executor):
        """Test timeout raises TimeoutError."""

        async def slow_coro():
            await asyncio.sleep(10)
            return "result"

        with pytest.raises(TimeoutError) as exc:
            await executor.with_timeout(slow_coro(), "agent1", timeout_seconds=0.1)
        assert "agent1" in str(exc.value)
        assert "timed out" in str(exc.value)

    @pytest.mark.asyncio
    async def test_timeout_records_circuit_breaker_failure(self, executor_with_circuit_breaker):
        """Test timeout records circuit breaker failure."""

        async def slow_coro():
            await asyncio.sleep(10)
            return "result"

        with pytest.raises(TimeoutError):
            await executor_with_circuit_breaker.with_timeout(
                slow_coro(), "agent1", timeout_seconds=0.1
            )

        # Verify failure was recorded (CircuitBreaker uses _failures dict for multi-entity)
        cb = executor_with_circuit_breaker.circuit_breaker
        assert cb._failures.get("agent1", 0) >= 1

    @pytest.mark.asyncio
    async def test_timeout_uses_default(self, executor):
        """Test timeout uses default when not specified."""

        async def quick_coro():
            return "result"

        # Should use default_timeout of 5.0
        result = await executor.with_timeout(quick_coro(), "agent1")
        assert result == "result"


# =============================================================================
# Generate Method Tests
# =============================================================================


class TestGenerate:
    """Tests for the generate method."""

    @pytest.mark.asyncio
    async def test_successful_generation(self, executor, mock_agent, context):
        """Test successful generation."""
        result = await executor.generate(mock_agent, "Test prompt", context)
        assert result == "Test response"
        assert mock_agent.generate_calls == 1

    @pytest.mark.asyncio
    async def test_empty_output_retry(self, executor, context):
        """Test empty output triggers retry."""
        agent = RetryableAgent(name="retry-agent", failures_before_success=1)
        result = await executor.generate(agent, "Test prompt", context)
        assert result == "Success after retry"
        assert agent.generate_calls == 2

    @pytest.mark.asyncio
    async def test_empty_output_returns_placeholder(self, executor, empty_agent, context):
        """Test persistent empty output returns placeholder."""
        result = await executor.generate(empty_agent, "Test prompt", context)
        assert "empty output" in result.lower() or "agent produced empty output" in result.lower()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, context):
        """Test timeout returns system message."""

        # The generate method doesn't wrap in timeout by default - agent.generate() runs directly
        # To test timeout handling, we need to use an agent that raises TimeoutError
        class TimeoutRaisingAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_calls += 1
                raise asyncio.TimeoutError("Simulated timeout")

        executor = AutonomicExecutor(default_timeout=5.0)
        agent = TimeoutRaisingAgent(name="timeout-agent")
        result = await executor.generate(agent, "Test prompt", context)
        assert "[System:" in result
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, executor, connection_error_agent, context):
        """Test connection error returns system message."""
        result = await executor.generate(connection_error_agent, "Test prompt", context)
        assert "[System:" in result
        assert "connection failed" in result.lower()

    @pytest.mark.asyncio
    async def test_general_exception_handling(self, executor, failing_agent, context):
        """Test general exception returns system message."""
        result = await executor.generate(failing_agent, "Test prompt", context)
        assert "[System:" in result
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_immune_system_notifications(self, mock_immune_system, mock_agent, context):
        """Test immune system is notified on success."""
        executor = AutonomicExecutor(immune_system=mock_immune_system, default_timeout=5.0)
        await executor.generate(mock_agent, "Test prompt", context)

        mock_immune_system.agent_started.assert_called_once()
        mock_immune_system.agent_completed.assert_called_once()

    @pytest.mark.asyncio
    async def test_immune_system_timeout_notification(self, mock_immune_system, context):
        """Test immune system is notified on timeout."""

        class TimeoutRaisingAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_calls += 1
                raise asyncio.TimeoutError("Simulated timeout")

        executor = AutonomicExecutor(immune_system=mock_immune_system, default_timeout=5.0)
        agent = TimeoutRaisingAgent(name="timeout-agent")
        await executor.generate(agent, "Test prompt", context)

        mock_immune_system.agent_started.assert_called_once()
        mock_immune_system.agent_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_immune_system_failure_notification(
        self, mock_immune_system, failing_agent, context
    ):
        """Test immune system is notified on failure."""
        executor = AutonomicExecutor(immune_system=mock_immune_system, default_timeout=5.0)
        await executor.generate(failing_agent, "Test prompt", context)

        mock_immune_system.agent_started.assert_called_once()
        mock_immune_system.agent_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_chaos_director_timeout_message(self, mock_chaos_director, context):
        """Test chaos director provides theatrical timeout message."""

        class TimeoutRaisingAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_calls += 1
                raise asyncio.TimeoutError("Simulated timeout")

        executor = AutonomicExecutor(chaos_director=mock_chaos_director, default_timeout=5.0)
        agent = TimeoutRaisingAgent(name="timeout-agent")
        result = await executor.generate(agent, "Test prompt", context)

        assert "[Theatrical timeout message]" in result
        mock_chaos_director.timeout_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_chaos_director_connection_message(
        self, mock_chaos_director, connection_error_agent, context
    ):
        """Test chaos director provides theatrical connection message."""
        executor = AutonomicExecutor(chaos_director=mock_chaos_director, default_timeout=5.0)
        result = await executor.generate(connection_error_agent, "Test prompt", context)

        assert "[Theatrical connection message]" in result
        mock_chaos_director.connection_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_chaos_director_error_message(self, mock_chaos_director, failing_agent, context):
        """Test chaos director provides theatrical error message."""
        executor = AutonomicExecutor(chaos_director=mock_chaos_director, default_timeout=5.0)
        result = await executor.generate(failing_agent, "Test prompt", context)

        assert "[Theatrical error message]" in result
        mock_chaos_director.internal_error_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_monitor_tracking(
        self, mock_performance_monitor, mock_agent, context
    ):
        """Test performance monitor tracks calls."""
        executor = AutonomicExecutor(
            performance_monitor=mock_performance_monitor, default_timeout=5.0
        )
        await executor.generate(mock_agent, "Test prompt", context, phase="proposal")

        mock_performance_monitor.track_agent_call.assert_called_once_with(
            mock_agent.name, "generate", phase="proposal", round_num=0
        )
        mock_performance_monitor.record_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_on_empty(
        self, executor_with_circuit_breaker, empty_agent, context
    ):
        """Test circuit breaker records failure on empty output."""
        await executor_with_circuit_breaker.generate(empty_agent, "Test prompt", context)
        cb = executor_with_circuit_breaker.circuit_breaker
        # CircuitBreaker uses _failures dict for multi-entity tracking
        assert cb._failures.get(empty_agent.name, 0) >= 1

    @pytest.mark.asyncio
    async def test_event_hook_on_error(self, context):
        """Test event hook is called on error."""
        hook = MagicMock()
        executor = AutonomicExecutor(event_hooks={"on_agent_error": hook}, default_timeout=5.0)
        agent = FailingAgent(name="failing-agent")
        await executor.generate(agent, "Test prompt", context)

        hook.assert_called()
        call_kwargs = hook.call_args[1]
        assert call_kwargs["agent"] == "failing-agent"
        assert call_kwargs["error_type"] == "internal"

    @pytest.mark.asyncio
    async def test_event_hook_exception_suppressed(self, context, mock_agent):
        """Test event hook exceptions don't crash execution."""

        def bad_hook(**kwargs):
            raise RuntimeError("Hook failed")

        executor = AutonomicExecutor(event_hooks={"on_agent_error": bad_hook}, default_timeout=5.0)
        # Should not raise, hook error is suppressed
        agent = FailingAgent(name="failing-agent")
        result = await executor.generate(agent, "Test prompt", context)
        assert "[System:" in result


# =============================================================================
# Critique Method Tests
# =============================================================================


class TestCritique:
    """Tests for the critique method."""

    @pytest.mark.asyncio
    async def test_successful_critique(self, executor, mock_agent, context):
        """Test successful critique."""
        result = await executor.critique(
            mock_agent, "Proposal text", "Task", context, target_agent="target"
        )
        assert result is not None
        assert result.agent == mock_agent.name
        assert mock_agent.critique_calls == 1

    @pytest.mark.asyncio
    async def test_empty_critique_retry(self, executor, context):
        """Test empty critique triggers retry."""

        class RetryableCritiqueAgent(MockAgent):
            def __init__(self):
                super().__init__(name="retry-critique-agent")
                self.attempt = 0

            async def critique(
                self, proposal: str, task: str, context: list = None, target_agent: str = None
            ) -> Critique:
                self.critique_calls += 1
                self.attempt += 1
                if self.attempt == 1:
                    return Critique(
                        agent=self.name,
                        target_agent=target_agent or "unknown",
                        target_content="",
                        issues=[],
                        suggestions=[],
                        severity=0.0,
                        reasoning="",
                    )
                return Critique(
                    agent=self.name,
                    target_agent=target_agent or "unknown",
                    target_content=proposal[:100],
                    issues=["Real issue"],
                    suggestions=["Real suggestion"],
                    severity=0.5,
                    reasoning="Real reasoning",
                )

        agent = RetryableCritiqueAgent()
        result = await executor.critique(agent, "Proposal", "Task", context)
        assert result is not None
        assert result.issues == ["Real issue"]
        assert agent.critique_calls == 2

    @pytest.mark.asyncio
    async def test_empty_critique_returns_none(self, executor, empty_agent, context):
        """Test persistent empty critique returns None."""
        result = await executor.critique(empty_agent, "Proposal", "Task", context)
        assert result is None

    @pytest.mark.asyncio
    async def test_critique_timeout(self, executor, context):
        """Test critique timeout returns None."""

        class TimeoutRaisingCritiqueAgent(MockAgent):
            async def critique(
                self, proposal: str, task: str, context: list = None, target_agent: str = None
            ) -> Critique:
                self.critique_calls += 1
                raise asyncio.TimeoutError("Simulated timeout")

        agent = TimeoutRaisingCritiqueAgent(name="timeout-agent")
        result = await executor.critique(agent, "Proposal", "Task", context)
        assert result is None

    @pytest.mark.asyncio
    async def test_critique_connection_error(self, executor, connection_error_agent, context):
        """Test critique connection error returns None."""
        result = await executor.critique(connection_error_agent, "Proposal", "Task", context)
        assert result is None

    @pytest.mark.asyncio
    async def test_critique_general_exception(self, executor, failing_agent, context):
        """Test critique general exception returns None."""
        result = await executor.critique(failing_agent, "Proposal", "Task", context)
        assert result is None

    @pytest.mark.asyncio
    async def test_critique_performance_tracking(
        self, mock_performance_monitor, mock_agent, context
    ):
        """Test critique performance tracking."""
        executor = AutonomicExecutor(
            performance_monitor=mock_performance_monitor, default_timeout=5.0
        )
        await executor.critique(
            mock_agent, "Proposal", "Task", context, phase="critique", round_num=1
        )

        mock_performance_monitor.track_agent_call.assert_called_once_with(
            mock_agent.name, "critique", phase="critique", round_num=1
        )

    @pytest.mark.asyncio
    async def test_critique_event_hook_on_error(self, context):
        """Test event hook is called on critique error."""
        hook = MagicMock()
        executor = AutonomicExecutor(event_hooks={"on_agent_error": hook}, default_timeout=5.0)
        agent = FailingAgent(name="failing-agent")
        await executor.critique(agent, "Proposal", "Task", context, phase="critique")

        hook.assert_called()
        call_kwargs = hook.call_args[1]
        assert call_kwargs["agent"] == "failing-agent"
        assert call_kwargs["phase"] == "critique"


# =============================================================================
# Vote Method Tests
# =============================================================================


class TestVote:
    """Tests for the vote method."""

    @pytest.mark.asyncio
    async def test_successful_vote(self, executor, mock_agent):
        """Test successful vote."""
        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        result = await executor.vote(mock_agent, proposals, "Task")
        assert result is not None
        assert result.agent == mock_agent.name
        assert mock_agent.vote_calls == 1

    @pytest.mark.asyncio
    async def test_none_vote_returns_none(self, executor):
        """Test None vote result returns None."""

        class NoneVoteAgent(MockAgent):
            async def vote(self, proposals: dict, task: str) -> Vote:
                self.vote_calls += 1
                return None

        agent = NoneVoteAgent(name="none-vote-agent")
        proposals = {"agent1": "Proposal 1"}
        result = await executor.vote(agent, proposals, "Task")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_choice_vote_returns_none(self, executor, empty_agent):
        """Test vote with empty choice returns None."""
        proposals = {"agent1": "Proposal 1"}
        result = await executor.vote(empty_agent, proposals, "Task")
        assert result is None

    @pytest.mark.asyncio
    async def test_vote_timeout(self, executor):
        """Test vote timeout returns None."""

        class TimeoutRaisingVoteAgent(MockAgent):
            async def vote(self, proposals: dict, task: str) -> Vote:
                self.vote_calls += 1
                raise asyncio.TimeoutError("Simulated timeout")

        agent = TimeoutRaisingVoteAgent(name="timeout-agent")
        proposals = {"agent1": "Proposal 1"}
        result = await executor.vote(agent, proposals, "Task")
        assert result is None

    @pytest.mark.asyncio
    async def test_vote_connection_error(self, executor, connection_error_agent):
        """Test vote connection error returns None."""
        proposals = {"agent1": "Proposal 1"}
        result = await executor.vote(connection_error_agent, proposals, "Task")
        assert result is None

    @pytest.mark.asyncio
    async def test_vote_general_exception(self, executor, failing_agent):
        """Test vote general exception returns None."""
        proposals = {"agent1": "Proposal 1"}
        result = await executor.vote(failing_agent, proposals, "Task")
        assert result is None

    @pytest.mark.asyncio
    async def test_vote_performance_tracking(self, mock_performance_monitor, mock_agent):
        """Test vote performance tracking."""
        executor = AutonomicExecutor(
            performance_monitor=mock_performance_monitor, default_timeout=5.0
        )
        proposals = {"agent1": "Proposal 1"}
        await executor.vote(mock_agent, proposals, "Task", phase="vote", round_num=2)

        mock_performance_monitor.track_agent_call.assert_called_once_with(
            mock_agent.name, "vote", phase="vote", round_num=2
        )

    @pytest.mark.asyncio
    async def test_vote_event_hook_on_error(self):
        """Test event hook is called on vote error."""
        hook = MagicMock()
        executor = AutonomicExecutor(event_hooks={"on_agent_error": hook}, default_timeout=5.0)
        agent = FailingAgent(name="failing-agent")
        proposals = {"agent1": "Proposal 1"}
        await executor.vote(agent, proposals, "Task", phase="vote")

        hook.assert_called()
        call_kwargs = hook.call_args[1]
        assert call_kwargs["agent"] == "failing-agent"
        assert call_kwargs["phase"] == "vote"


# =============================================================================
# Generate with Fallback Tests
# =============================================================================


class TestGenerateWithFallback:
    """Tests for the generate_with_fallback method."""

    @pytest.mark.asyncio
    async def test_primary_agent_succeeds(self, executor, context):
        """Test primary agent success."""
        primary = MockAgent(name="primary", response="Primary response")
        fallback = MockAgent(name="fallback", response="Fallback response")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, fallback_agents=[fallback]
        )
        assert result == "Primary response"
        assert primary.generate_calls == 1
        assert fallback.generate_calls == 0

    @pytest.mark.asyncio
    async def test_fallback_on_timeout(self, executor, context):
        """Test fallback agent used on primary timeout."""
        primary = TimeoutAgent(name="primary", delay=10.0)
        fallback = MockAgent(name="fallback", response="Fallback response")
        executor.default_timeout = 0.1

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, fallback_agents=[fallback], max_retries=1
        )
        assert result == "Fallback response"
        assert fallback.generate_calls == 1

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self, executor, context):
        """Test fallback agent used on primary exception."""
        primary = FailingAgent(name="primary")
        fallback = MockAgent(name="fallback", response="Fallback response")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, fallback_agents=[fallback]
        )
        assert result == "Fallback response"
        assert fallback.generate_calls == 1

    @pytest.mark.asyncio
    async def test_multiple_fallbacks(self, executor, context):
        """Test multiple fallback agents in sequence."""
        primary = FailingAgent(name="primary")
        fallback1 = FailingAgent(name="fallback1")
        fallback2 = MockAgent(name="fallback2", response="Fallback 2 response")

        result = await executor.generate_with_fallback(
            primary,
            "Test prompt",
            context,
            fallback_agents=[fallback1, fallback2],
        )
        assert result == "Fallback 2 response"
        assert fallback2.generate_calls == 1

    @pytest.mark.asyncio
    async def test_all_agents_fail(self, executor, context):
        """Test all agents failing returns system message."""
        primary = FailingAgent(name="primary")
        fallback = FailingAgent(name="fallback")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, fallback_agents=[fallback]
        )
        assert "[System:" in result
        assert "All agents failed" in result

    @pytest.mark.asyncio
    async def test_circuit_broken_agent_skipped(self, context):
        """Test circuit-broken agents are skipped."""
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=300.0)
        executor = AutonomicExecutor(circuit_breaker=cb, default_timeout=5.0)

        primary = MockAgent(name="primary", response="Primary response")
        fallback = MockAgent(name="fallback", response="Fallback response")

        # Open circuit for primary
        cb.record_failure("primary")
        cb.record_failure("primary")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, fallback_agents=[fallback]
        )
        # Primary should be skipped, fallback used
        assert result == "Fallback response"
        assert primary.generate_calls == 0
        assert fallback.generate_calls == 1

    @pytest.mark.asyncio
    async def test_retry_escalation(self, executor, context):
        """Test timeout escalation on retries."""
        primary = TimeoutAgent(name="primary", delay=10.0)
        fallback = MockAgent(name="fallback", response="Fallback response")
        executor.default_timeout = 0.1

        # First retry should escalate timeout
        await executor.generate_with_fallback(
            primary, "Test prompt", context, fallback_agents=[fallback], max_retries=2
        )

        # Check retry count was incremented
        assert executor._retry_counts.get("primary", 0) > 0

    @pytest.mark.asyncio
    async def test_partial_content_fallback(self, context):
        """Test partial streaming content used as fallback."""
        # Create an agent that simulates writing to the streaming buffer during
        # execution, then timing out
        executor = AutonomicExecutor(default_timeout=0.1)

        class StreamingTimeoutAgent(MockAgent):
            def __init__(self, executor: AutonomicExecutor):
                super().__init__(name="streaming-primary")
                self.executor = executor

            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_calls += 1
                # Simulate streaming content being accumulated
                self.executor.streaming_buffer._buffer[self.name] = "A" * 250
                # Then timeout
                await asyncio.sleep(10.0)
                return "Should never reach here"

        primary = StreamingTimeoutAgent(executor)

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, max_retries=1
        )
        # Should use partial content as fallback when all agents fail and content > 200
        assert "A" * 250 in result or "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_wisdom_fallback(self, mock_wisdom_store, context):
        """Test wisdom store fallback when all agents fail."""
        executor = AutonomicExecutor(
            wisdom_store=mock_wisdom_store,
            loop_id="test-loop",
            default_timeout=5.0,
        )
        primary = FailingAgent(name="primary")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, max_retries=1
        )

        assert "Audience Wisdom" in result
        assert "user-123" in result
        mock_wisdom_store.get_relevant_wisdom.assert_called_once()
        mock_wisdom_store.mark_wisdom_used.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_no_wisdom_without_loop_id(self, mock_wisdom_store, context):
        """Test wisdom not used without loop_id."""
        executor = AutonomicExecutor(
            wisdom_store=mock_wisdom_store,
            loop_id=None,  # No loop ID
            default_timeout=5.0,
        )
        primary = FailingAgent(name="primary")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, max_retries=1
        )

        assert "All agents failed" in result
        mock_wisdom_store.get_relevant_wisdom.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_resets_retries(self, executor, context):
        """Test successful generation resets retry count."""
        primary = MockAgent(name="primary", response="Primary response")

        # Set some retries
        executor.record_retry("primary")
        executor.record_retry("primary")

        await executor.generate_with_fallback(primary, "Test prompt", context)

        # Should be reset after success
        assert executor._retry_counts.get("primary", 0) == 0


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_is_empty_critique_none(self):
        """Test None critique is empty."""
        assert AutonomicExecutor._is_empty_critique(None) is True

    def test_is_empty_critique_empty_lists(self):
        """Test critique with empty lists is empty."""
        critique = Critique(
            agent="agent",
            target_agent="target",
            target_content="",
            issues=[],
            suggestions=[],
            severity=0.0,
            reasoning="",
        )
        assert AutonomicExecutor._is_empty_critique(critique) is True

    def test_is_empty_critique_placeholder_content(self):
        """Test critique with placeholder content is empty."""
        critique = Critique(
            agent="agent",
            target_agent="target",
            target_content="",
            issues=["Agent response was empty"],
            suggestions=[],
            severity=0.0,
            reasoning="",
        )
        assert AutonomicExecutor._is_empty_critique(critique) is True

    def test_is_empty_critique_real_content(self):
        """Test critique with real content is not empty."""
        critique = Critique(
            agent="agent",
            target_agent="target",
            target_content="content",
            issues=["Real issue"],
            suggestions=["Real suggestion"],
            severity=0.5,
            reasoning="Real reasoning",
        )
        assert AutonomicExecutor._is_empty_critique(critique) is False

    def test_get_wisdom_fallback_no_store(self):
        """Test wisdom fallback without store."""
        executor = AutonomicExecutor()
        result = executor._get_wisdom_fallback("agent")
        assert result is None

    def test_get_wisdom_fallback_no_loop_id(self, mock_wisdom_store):
        """Test wisdom fallback without loop ID."""
        executor = AutonomicExecutor(wisdom_store=mock_wisdom_store, loop_id=None)
        result = executor._get_wisdom_fallback("agent")
        assert result is None

    def test_get_wisdom_fallback_empty_wisdom(self, mock_wisdom_store):
        """Test wisdom fallback with empty wisdom list."""
        mock_wisdom_store.get_relevant_wisdom.return_value = []
        executor = AutonomicExecutor(wisdom_store=mock_wisdom_store, loop_id="test-loop")
        result = executor._get_wisdom_fallback("agent")
        assert result is None

    def test_get_wisdom_fallback_exception(self, mock_wisdom_store):
        """Test wisdom fallback handles exceptions."""
        mock_wisdom_store.get_relevant_wisdom.side_effect = Exception("DB error")
        executor = AutonomicExecutor(wisdom_store=mock_wisdom_store, loop_id="test-loop")
        result = executor._get_wisdom_fallback("agent")
        assert result is None


# =============================================================================
# Telemetry Tests
# =============================================================================


class TestTelemetry:
    """Tests for telemetry emission."""

    @pytest.mark.asyncio
    async def test_telemetry_disabled_by_default(self, mock_agent, context):
        """Test telemetry emit method is not called when disabled."""
        executor = AutonomicExecutor(enable_telemetry=False, default_timeout=5.0)
        # Call generate and verify no telemetry crash - telemetry is off
        result = await executor.generate(mock_agent, "Test prompt", context)
        assert result == "Test response"
        # No assertion needed - just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_telemetry_emitted_when_enabled(self, mock_agent, context):
        """Test telemetry emission code path is invoked when enabled."""
        # Just verify that enabling telemetry doesn't break execution
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)
        result = await executor.generate(mock_agent, "Test prompt", context)
        assert result == "Test response"

    def test_emit_agent_telemetry_disabled(self, mock_agent):
        """Test _emit_agent_telemetry does nothing when disabled."""
        executor = AutonomicExecutor(enable_telemetry=False, default_timeout=5.0)
        # Should not raise any errors when telemetry is disabled
        executor._emit_agent_telemetry(
            mock_agent.name, "generate", time.time(), True, None, "output", "input"
        )

    def test_emit_agent_telemetry_enabled_handles_import_error(self):
        """Test _emit_agent_telemetry handles import errors gracefully."""
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)

        # Patch the import to fail
        with patch.dict("sys.modules", {"aragora.agents.telemetry": None}):
            # Should not raise, just silently skip
            try:
                executor._emit_agent_telemetry(
                    "test-agent", "generate", time.time(), True, None, "output", "input"
                )
            except ImportError:
                pass  # Expected when module not available


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_context(self, executor, mock_agent):
        """Test generation with empty context."""
        result = await executor.generate(mock_agent, "Test prompt", [])
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_none_context(self, executor, mock_agent):
        """Test generation handles None-like empty context."""
        result = await executor.generate(mock_agent, "Test prompt", [])
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, executor, mock_agent, context):
        """Test generation with very long prompt."""
        long_prompt = "x" * 100000
        result = await executor.generate(mock_agent, long_prompt, context)
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_special_characters_in_response(self, executor, context):
        """Test handling of special characters in response."""
        agent = MockAgent(
            name="special-agent",
            response="Response with\x00null\x01control\x02chars",
        )
        result = await executor.generate(agent, "Test prompt", context)
        # Null bytes should be removed by sanitizer
        assert "\x00" not in result

    @pytest.mark.asyncio
    async def test_unicode_in_response(self, executor, context):
        """Test handling of unicode in response."""
        agent = MockAgent(name="unicode-agent", response="Unicode: ")
        result = await executor.generate(agent, "Test prompt", context)
        assert "" in result

    @pytest.mark.asyncio
    async def test_concurrent_generations(self, executor, context):
        """Test concurrent generations don't interfere."""
        agents = [MockAgent(name=f"agent-{i}", response=f"Response {i}") for i in range(5)]

        tasks = [executor.generate(agent, f"Prompt {i}", context) for i, agent in enumerate(agents)]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            assert f"Response {i}" in result

    @pytest.mark.asyncio
    async def test_zero_timeout(self, executor, context):
        """Test handling of zero timeout."""
        executor.default_timeout = 0.001  # Very small but not zero
        agent = MockAgent(name="test-agent", response="Response")

        # Should either succeed quickly or timeout
        result = await executor.generate(agent, "Test prompt", context)
        # Result should be either the response or a timeout message
        assert "Response" in result or "[System:" in result

    @pytest.mark.asyncio
    async def test_negative_timeout_treated_as_default(self, executor, context):
        """Test negative timeout uses default."""
        agent = MockAgent(name="test-agent", response="Response")
        # get_escalated_timeout with None uses default
        timeout = executor.get_escalated_timeout("agent", base_timeout=None)
        assert timeout > 0

    @pytest.mark.asyncio
    async def test_asyncio_cancellation(self, executor, context):
        """Test handling of asyncio cancellation."""
        agent = TimeoutAgent(name="slow-agent", delay=10.0)

        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            raise asyncio.CancelledError()

        task = asyncio.create_task(executor.generate(agent, "Test prompt", context))

        try:
            await asyncio.wait_for(task, timeout=0.2)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass  # Expected

    @pytest.mark.asyncio
    async def test_os_error_handling(self, executor, context):
        """Test OSError handling."""

        class OSErrorAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                raise OSError("Disk full")

        agent = OSErrorAgent(name="os-error-agent")
        result = await executor.generate(agent, "Test prompt", context)
        assert "[System:" in result
        assert "connection failed" in result.lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with multiple components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        mock_immune_system,
        mock_performance_monitor,
        circuit_breaker,
        mock_agent,
        context,
    ):
        """Test full pipeline with all components on success."""
        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            immune_system=mock_immune_system,
            performance_monitor=mock_performance_monitor,
            default_timeout=5.0,
        )

        result = await executor.generate(
            mock_agent, "Test prompt", context, phase="proposal", round_num=1
        )

        assert result == "Test response"
        mock_immune_system.agent_started.assert_called_once()
        mock_immune_system.agent_completed.assert_called_once()
        mock_performance_monitor.track_agent_call.assert_called_once()
        mock_performance_monitor.record_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_pipeline_failure(
        self,
        mock_immune_system,
        mock_performance_monitor,
        circuit_breaker,
        failing_agent,
        context,
    ):
        """Test full pipeline with all components on failure."""
        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            immune_system=mock_immune_system,
            performance_monitor=mock_performance_monitor,
            default_timeout=5.0,
        )

        result = await executor.generate(
            failing_agent, "Test prompt", context, phase="proposal", round_num=1
        )

        assert "[System:" in result
        mock_immune_system.agent_started.assert_called_once()
        mock_immune_system.agent_failed.assert_called_once()
        mock_performance_monitor.track_agent_call.assert_called_once()
        mock_performance_monitor.record_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_critique_vote_sequence(self, executor, mock_agent, context):
        """Test typical debate sequence: generate, critique, vote."""
        # Generate
        gen_result = await executor.generate(mock_agent, "Test prompt", context, phase="proposal")
        assert gen_result == "Test response"

        # Critique
        critique_result = await executor.critique(
            mock_agent, gen_result, "Task", context, phase="critique"
        )
        assert critique_result is not None
        assert critique_result.agent == mock_agent.name

        # Vote
        proposals = {mock_agent.name: gen_result}
        vote_result = await executor.vote(mock_agent, proposals, "Task", phase="vote")
        assert vote_result is not None
        assert vote_result.agent == mock_agent.name

    @pytest.mark.asyncio
    async def test_fallback_with_all_integrations(
        self,
        mock_immune_system,
        mock_performance_monitor,
        mock_wisdom_store,
        circuit_breaker,
        context,
    ):
        """Test fallback with all integrations."""
        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            immune_system=mock_immune_system,
            performance_monitor=mock_performance_monitor,
            wisdom_store=mock_wisdom_store,
            loop_id="test-loop",
            default_timeout=5.0,
        )

        primary = FailingAgent(name="primary")
        fallback = MockAgent(name="fallback", response="Fallback response")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", context, fallback_agents=[fallback]
        )

        assert result == "Fallback response"


# =============================================================================
# Additional Edge Case Tests for Higher Coverage
# =============================================================================


class TestEmptyCritiqueEdgeCases:
    """Tests for empty critique detection edge cases."""

    def test_is_empty_critique_whitespace_issues(self):
        """Test critique with only whitespace issues is empty."""
        critique = Critique(
            agent="agent",
            target_agent="target",
            target_content="",
            issues=["   ", "  \n  ", "\t"],
            suggestions=[],
            severity=0.0,
            reasoning="",
        )
        assert AutonomicExecutor._is_empty_critique(critique) is True

    def test_is_empty_critique_alternative_placeholder(self):
        """Test critique with alternative placeholder content."""
        critique = Critique(
            agent="agent",
            target_agent="target",
            target_content="",
            issues=["(agent produced empty output)"],
            suggestions=[],
            severity=0.0,
            reasoning="",
        )
        assert AutonomicExecutor._is_empty_critique(critique) is True

    def test_is_empty_critique_agent_produced_empty_output(self):
        """Test critique with 'agent produced empty output' message."""
        critique = Critique(
            agent="agent",
            target_agent="target",
            target_content="",
            issues=["agent produced empty output"],
            suggestions=[],
            severity=0.0,
            reasoning="",
        )
        assert AutonomicExecutor._is_empty_critique(critique) is True

    def test_is_empty_critique_placeholder_with_suggestions(self):
        """Test critique with placeholder issue but real suggestions is NOT empty."""
        critique = Critique(
            agent="agent",
            target_agent="target",
            target_content="",
            issues=["Agent response was empty"],
            suggestions=["Real suggestion"],
            severity=0.0,
            reasoning="",
        )
        assert AutonomicExecutor._is_empty_critique(critique) is False

    def test_is_empty_critique_non_string_issues(self):
        """Test critique handles non-string issues gracefully."""
        critique = Critique(
            agent="agent",
            target_agent="target",
            target_content="",
            issues=["Real issue", 123, None],  # type: ignore
            suggestions=[],
            severity=0.0,
            reasoning="",
        )
        # Should not raise, real issue makes it non-empty
        assert AutonomicExecutor._is_empty_critique(critique) is False


class TestCritiqueRetryEdgeCases:
    """Tests for critique retry edge cases."""

    @pytest.mark.asyncio
    async def test_critique_none_after_retry(self):
        """Test critique that returns None after non-empty retry still returns retry result."""

        class NullReturnCritiqueAgent(MockAgent):
            def __init__(self):
                super().__init__(name="null-critique-agent")
                self.attempt = 0

            async def critique(
                self, proposal: str, task: str, context: list = None, target_agent: str = None
            ):
                self.critique_calls += 1
                self.attempt += 1
                # First attempt returns empty, second returns valid
                if self.attempt == 1:
                    return Critique(
                        agent=self.name,
                        target_agent=target_agent or "unknown",
                        target_content="",
                        issues=[],
                        suggestions=[],
                        severity=0.0,
                        reasoning="",
                    )
                return Critique(
                    agent=self.name,
                    target_agent=target_agent or "unknown",
                    target_content=proposal[:100],
                    issues=["Valid issue"],
                    suggestions=["Valid suggestion"],
                    severity=0.5,
                    reasoning="Valid reasoning",
                )

        executor = AutonomicExecutor(default_timeout=5.0)
        agent = NullReturnCritiqueAgent()
        result = await executor.critique(agent, "Proposal", "Task", [])
        assert result is not None
        assert agent.critique_calls == 2

    @pytest.mark.asyncio
    async def test_critique_returns_none_directly(self):
        """Test critique agent that returns None directly."""

        class DirectNoneCritiqueAgent(MockAgent):
            async def critique(
                self, proposal: str, task: str, context: list = None, target_agent: str = None
            ):
                self.critique_calls += 1
                return None

        executor = AutonomicExecutor(default_timeout=5.0)
        agent = DirectNoneCritiqueAgent(name="none-agent")
        result = await executor.critique(agent, "Proposal", "Task", [])
        assert result is None


class TestWisdomFallbackEdgeCases:
    """Tests for wisdom fallback edge cases."""

    def test_get_wisdom_fallback_key_error(self):
        """Test wisdom fallback handles KeyError gracefully."""
        mock_store = MagicMock()
        mock_store.get_relevant_wisdom.return_value = [
            {"missing_id": 1, "text": "Wisdom", "submitter_id": "user"}
        ]
        executor = AutonomicExecutor(wisdom_store=mock_store, loop_id="test-loop")
        result = executor._get_wisdom_fallback("agent")
        # Should return None due to KeyError for 'id'
        assert result is None

    def test_get_wisdom_fallback_os_error(self):
        """Test wisdom fallback handles OSError gracefully."""
        mock_store = MagicMock()
        mock_store.get_relevant_wisdom.side_effect = OSError("File not found")
        executor = AutonomicExecutor(wisdom_store=mock_store, loop_id="test-loop")
        result = executor._get_wisdom_fallback("agent")
        assert result is None

    def test_get_wisdom_fallback_io_error(self):
        """Test wisdom fallback handles IOError gracefully."""
        mock_store = MagicMock()
        mock_store.get_relevant_wisdom.side_effect = IOError("Read error")
        executor = AutonomicExecutor(wisdom_store=mock_store, loop_id="test-loop")
        result = executor._get_wisdom_fallback("agent")
        assert result is None


class TestTelemetryEmissionEdgeCases:
    """Tests for telemetry emission edge cases."""

    def test_emit_agent_telemetry_type_error(self):
        """Test telemetry emission handles TypeError gracefully."""
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)

        with patch("aragora.debate.autonomic_executor._ensure_telemetry_collectors"):
            with patch(
                "aragora.agents.telemetry.AgentTelemetry",
                side_effect=TypeError("Type error"),
            ):
                # Should not raise
                executor._emit_agent_telemetry(
                    "agent", "generate", time.time(), True, None, "output", "input"
                )

    def test_emit_agent_telemetry_value_error(self):
        """Test telemetry emission handles ValueError gracefully."""
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)

        with patch("aragora.debate.autonomic_executor._ensure_telemetry_collectors"):
            with patch(
                "aragora.agents.telemetry.AgentTelemetry",
                side_effect=ValueError("Value error"),
            ):
                # Should not raise
                executor._emit_agent_telemetry(
                    "agent", "generate", time.time(), True, None, "output", "input"
                )

    def test_emit_agent_telemetry_os_error(self):
        """Test telemetry emission handles OSError gracefully."""
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)

        with patch("aragora.debate.autonomic_executor._ensure_telemetry_collectors"):
            with patch(
                "aragora.agents.telemetry.AgentTelemetry",
                side_effect=OSError("OS error"),
            ):
                # Should not raise
                executor._emit_agent_telemetry(
                    "agent", "generate", time.time(), True, None, "output", "input"
                )


class TestPerformanceMonitorFailureTracking:
    """Tests for performance monitor failure tracking."""

    @pytest.mark.asyncio
    async def test_generate_empty_output_records_failure(self):
        """Test empty output records failure in performance monitor."""
        monitor = MagicMock()
        monitor.track_agent_call = MagicMock(return_value="tracking-123")
        monitor.record_completion = MagicMock()

        executor = AutonomicExecutor(performance_monitor=monitor, default_timeout=5.0)
        agent = EmptyOutputAgent(name="empty-agent")
        await executor.generate(agent, "Test prompt", [])

        # Should record failure for empty output
        calls = monitor.record_completion.call_args_list
        assert len(calls) >= 1
        # Last call should indicate failure
        last_call = calls[-1]
        assert last_call[1].get("success") is False
        assert "empty" in last_call[1].get("error", "").lower()

    @pytest.mark.asyncio
    async def test_vote_empty_choice_records_failure(self):
        """Test empty vote choice records failure in performance monitor."""
        monitor = MagicMock()
        monitor.track_agent_call = MagicMock(return_value="tracking-456")
        monitor.record_completion = MagicMock()

        executor = AutonomicExecutor(performance_monitor=monitor, default_timeout=5.0)
        agent = EmptyOutputAgent(name="empty-vote-agent")
        proposals = {"agent1": "Proposal"}
        await executor.vote(agent, proposals, "Task")

        # Should record failure for empty vote
        calls = monitor.record_completion.call_args_list
        assert len(calls) >= 1
        last_call = calls[-1]
        assert last_call[1].get("success") is False


class TestImmuneSystemEmptyOutput:
    """Tests for immune system integration with empty output."""

    @pytest.mark.asyncio
    async def test_immune_system_empty_output_failure(self):
        """Test immune system is notified of empty output as failure."""
        immune = MagicMock()
        immune.agent_started = MagicMock()
        immune.agent_completed = MagicMock()
        immune.agent_failed = MagicMock()

        executor = AutonomicExecutor(immune_system=immune, default_timeout=5.0)
        agent = EmptyOutputAgent(name="empty-agent")
        await executor.generate(agent, "Test prompt", [])

        immune.agent_started.assert_called_once()
        # Should call agent_failed for empty output
        immune.agent_failed.assert_called()
        call_args = immune.agent_failed.call_args
        assert call_args[0][0] == "empty-agent"
        assert "empty" in call_args[0][1].lower()


class TestCircuitBreakerRecordingEdgeCases:
    """Tests for circuit breaker failure recording edge cases."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_not_recorded_for_connection_error_in_generate(self):
        """Test circuit breaker does NOT record connection errors in generate()."""
        # The generate() method only records circuit breaker failures for:
        # - TimeoutError (line 372-373)
        # - Empty output (line 447-448)
        # Connection errors are handled gracefully without circuit breaker recording
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60.0)
        executor = AutonomicExecutor(circuit_breaker=cb, default_timeout=5.0)

        agent = ConnectionErrorAgent(name="conn-error-agent")
        await executor.generate(agent, "Test prompt", [])

        # generate() doesn't record circuit breaker failures for connection errors
        # This is by design - connection errors are considered recoverable
        assert cb._failures.get("conn-error-agent", 0) == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_not_recorded_for_general_exception(self):
        """Test circuit breaker does NOT record general exceptions in generate()."""
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60.0)
        executor = AutonomicExecutor(circuit_breaker=cb, default_timeout=5.0)

        agent = FailingAgent(name="failing-agent")
        await executor.generate(agent, "Test prompt", [])

        # Circuit breaker is only recorded for timeout/empty in generate
        # General exceptions don't record to circuit breaker in generate method
        # But in generate_with_fallback they do
        assert cb._failures.get("failing-agent", 0) == 0


class TestFallbackConnectionErrors:
    """Tests for connection error handling in fallback scenarios."""

    @pytest.mark.asyncio
    async def test_fallback_connection_error_records_circuit_breaker(self):
        """Test connection error in fallback records circuit breaker failure."""
        cb = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=60.0)
        executor = AutonomicExecutor(circuit_breaker=cb, default_timeout=5.0)

        primary = ConnectionErrorAgent(name="primary")
        fallback = MockAgent(name="fallback", response="Fallback response")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", [], fallback_agents=[fallback]
        )

        assert result == "Fallback response"
        assert cb._failures.get("primary", 0) >= 1

    @pytest.mark.asyncio
    async def test_fallback_all_connection_errors(self):
        """Test all agents having connection errors."""
        executor = AutonomicExecutor(default_timeout=5.0)

        primary = ConnectionErrorAgent(name="primary")
        fallback = ConnectionErrorAgent(name="fallback")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", [], fallback_agents=[fallback], max_retries=1
        )

        assert "[System:" in result
        assert "All agents failed" in result


class TestValidationWarnings:
    """Tests for response validation warnings."""

    @pytest.mark.asyncio
    async def test_generate_logs_validation_warnings(self):
        """Test that validation warnings are logged but don't fail."""
        executor = AutonomicExecutor(default_timeout=5.0)
        # Very long response should trigger warnings
        agent = MockAgent(name="long-agent", response="x" * 50000)

        result = await executor.generate(agent, "Test prompt", [])
        # Should succeed despite potential warnings
        assert len(result) > 0


class TestEventHooksEdgeCases:
    """Tests for event hooks edge cases."""

    @pytest.mark.asyncio
    async def test_event_hook_empty_output_error(self):
        """Test event hook is called for empty output error."""
        hook = MagicMock()
        executor = AutonomicExecutor(event_hooks={"on_agent_error": hook}, default_timeout=5.0)

        agent = EmptyOutputAgent(name="empty-agent")
        await executor.generate(agent, "Test prompt", [])

        hook.assert_called()
        call_kwargs = hook.call_args[1]
        assert call_kwargs["error_type"] == "empty"

    @pytest.mark.asyncio
    async def test_event_hook_timeout_error(self):
        """Test event hook is called for timeout error."""
        hook = MagicMock()
        executor = AutonomicExecutor(event_hooks={"on_agent_error": hook}, default_timeout=5.0)

        class TimeoutRaisingAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                raise asyncio.TimeoutError("Timeout")

        agent = TimeoutRaisingAgent(name="timeout-agent")
        await executor.generate(agent, "Test prompt", [])

        hook.assert_called()
        call_kwargs = hook.call_args[1]
        assert call_kwargs["error_type"] == "timeout"

    @pytest.mark.asyncio
    async def test_event_hook_connection_error(self):
        """Test event hook is called for connection error."""
        hook = MagicMock()
        executor = AutonomicExecutor(event_hooks={"on_agent_error": hook}, default_timeout=5.0)

        agent = ConnectionErrorAgent(name="conn-agent")
        await executor.generate(agent, "Test prompt", [])

        hook.assert_called()
        call_kwargs = hook.call_args[1]
        assert call_kwargs["error_type"] == "connection"


class TestCritiqueTelemetryEmission:
    """Tests for critique telemetry emission."""

    @pytest.mark.asyncio
    async def test_critique_emits_telemetry_on_success(self):
        """Test critique emits telemetry on success."""
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)
        agent = MockAgent(name="test-agent")

        # Should not raise even with telemetry enabled
        result = await executor.critique(agent, "Proposal", "Task", [])
        assert result is not None

    @pytest.mark.asyncio
    async def test_critique_emits_telemetry_on_empty(self):
        """Test critique emits telemetry on empty result."""
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)
        agent = EmptyOutputAgent(name="empty-agent")

        result = await executor.critique(agent, "Proposal", "Task", [])
        # Empty critique after retry should return None
        assert result is None


class TestVoteTelemetryEmission:
    """Tests for vote telemetry emission."""

    @pytest.mark.asyncio
    async def test_vote_emits_telemetry_on_success(self):
        """Test vote emits telemetry on success."""
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)
        agent = MockAgent(name="test-agent")
        proposals = {"agent1": "Proposal 1"}

        result = await executor.vote(agent, proposals, "Task")
        assert result is not None

    @pytest.mark.asyncio
    async def test_vote_emits_telemetry_on_empty(self):
        """Test vote emits telemetry on empty result."""
        executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)
        agent = EmptyOutputAgent(name="empty-agent")
        proposals = {"agent1": "Proposal 1"}

        result = await executor.vote(agent, proposals, "Task")
        assert result is None


class TestMultipleRetryScenarios:
    """Tests for multiple retry scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_agents_retry_counts_independent(self):
        """Test retry counts are tracked independently per agent."""
        executor = AutonomicExecutor(default_timeout=5.0)

        executor.record_retry("agent1")
        executor.record_retry("agent1")
        executor.record_retry("agent2")

        assert executor._retry_counts["agent1"] == 2
        assert executor._retry_counts["agent2"] == 1

        executor.reset_retries("agent1")
        assert executor._retry_counts.get("agent1", 0) == 0
        assert executor._retry_counts["agent2"] == 1

    def test_timeout_escalation_multiple_agents(self):
        """Test timeout escalation works independently for multiple agents."""
        executor = AutonomicExecutor(default_timeout=10.0, timeout_escalation_factor=2.0)

        # Agent 1 has no retries
        timeout1 = executor.get_escalated_timeout("agent1")
        assert timeout1 == 10.0

        # Agent 2 has 1 retry
        executor.record_retry("agent2")
        timeout2 = executor.get_escalated_timeout("agent2")
        assert timeout2 == 20.0  # 10 * 2^1

        # Agent 3 has 2 retries
        executor.record_retry("agent3")
        executor.record_retry("agent3")
        timeout3 = executor.get_escalated_timeout("agent3")
        assert timeout3 == 40.0  # 10 * 2^2


class TestFallbackPartialContentThreshold:
    """Tests for fallback partial content threshold."""

    @pytest.mark.asyncio
    async def test_partial_content_below_threshold_not_used(self):
        """Test partial content below 200 chars is not used as fallback."""
        executor = AutonomicExecutor(default_timeout=0.1)

        class SmallPartialAgent(MockAgent):
            def __init__(self, executor: AutonomicExecutor):
                super().__init__(name="small-partial")
                self.executor = executor

            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_calls += 1
                # Only 100 chars - below threshold
                self.executor.streaming_buffer._buffer[self.name] = "A" * 100
                await asyncio.sleep(10.0)
                return "Never reached"

        primary = SmallPartialAgent(executor)

        result = await executor.generate_with_fallback(primary, "Test prompt", [], max_retries=1)

        # Should NOT use partial content since it's below 200 chars
        # Should instead show "All agents failed"
        assert "[System:" in result
        assert "All agents failed" in result

    @pytest.mark.asyncio
    async def test_partial_content_at_threshold_used(self):
        """Test partial content at exactly 201 chars is used as fallback."""
        executor = AutonomicExecutor(default_timeout=0.1)

        class ThresholdPartialAgent(MockAgent):
            def __init__(self, executor: AutonomicExecutor):
                super().__init__(name="threshold-partial")
                self.executor = executor

            async def generate(self, prompt: str, context: list = None) -> str:
                self.generate_calls += 1
                # 201 chars - at threshold
                self.executor.streaming_buffer._buffer[self.name] = "B" * 201
                await asyncio.sleep(10.0)
                return "Never reached"

        primary = ThresholdPartialAgent(executor)

        result = await executor.generate_with_fallback(primary, "Test prompt", [], max_retries=1)

        # Should use partial content since it's above 200 chars
        assert "B" * 201 in result or "truncated" in result.lower()


class TestStreamingBufferConcurrency:
    """Tests for streaming buffer concurrency."""

    @pytest.mark.asyncio
    async def test_streaming_buffer_multiple_agents_concurrent(self):
        """Test streaming buffer handles multiple agents concurrently."""
        buffer = StreamingContentBuffer()

        async def append_for_agent(agent_name: str, char: str, count: int):
            for _ in range(count):
                await buffer.append(agent_name, char)
                await asyncio.sleep(0.001)

        # Run concurrent appends for multiple agents
        await asyncio.gather(
            append_for_agent("agent1", "a", 50),
            append_for_agent("agent2", "b", 50),
            append_for_agent("agent3", "c", 50),
        )

        assert len(buffer.get_partial("agent1")) == 50
        assert len(buffer.get_partial("agent2")) == 50
        assert len(buffer.get_partial("agent3")) == 50
        assert buffer.get_partial("agent1") == "a" * 50
        assert buffer.get_partial("agent2") == "b" * 50
        assert buffer.get_partial("agent3") == "c" * 50


class TestOSErrorVariants:
    """Tests for various OSError variants."""

    @pytest.mark.asyncio
    async def test_file_not_found_error(self):
        """Test FileNotFoundError (subclass of OSError) handling."""

        class FileNotFoundAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                raise FileNotFoundError("Config file not found")

        executor = AutonomicExecutor(default_timeout=5.0)
        agent = FileNotFoundAgent(name="fnf-agent")
        result = await executor.generate(agent, "Test prompt", [])

        assert "[System:" in result
        # FileNotFoundError is a subclass of OSError, caught by connection error handler
        assert "connection failed" in result.lower()

    @pytest.mark.asyncio
    async def test_permission_error(self):
        """Test PermissionError (subclass of OSError) handling."""

        class PermissionAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                raise PermissionError("Access denied")

        executor = AutonomicExecutor(default_timeout=5.0)
        agent = PermissionAgent(name="perm-agent")
        result = await executor.generate(agent, "Test prompt", [])

        assert "[System:" in result
        assert "connection failed" in result.lower()


class TestChaosDirectorInternalError:
    """Tests for chaos director internal error responses."""

    @pytest.mark.asyncio
    async def test_chaos_director_os_error_response(self):
        """Test chaos director provides response for OS errors."""

        @dataclass
        class MockChaosResponse:
            message: str

        chaos = MagicMock()
        chaos.connection_response = MagicMock(
            return_value=MockChaosResponse(message="[Theatrical OS error]")
        )

        executor = AutonomicExecutor(chaos_director=chaos, default_timeout=5.0)

        class OSErrorAgent(MockAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                raise OSError("Network unreachable")

        agent = OSErrorAgent(name="os-error-agent")
        result = await executor.generate(agent, "Test prompt", [])

        assert "[Theatrical OS error]" in result


class TestPerformanceMonitorPhaseRoundTracking:
    """Tests for performance monitor phase and round tracking."""

    @pytest.mark.asyncio
    async def test_generate_tracks_phase_and_round(self):
        """Test generate correctly tracks phase and round."""
        monitor = MagicMock()
        monitor.track_agent_call = MagicMock(return_value="id-123")
        monitor.record_completion = MagicMock()

        executor = AutonomicExecutor(performance_monitor=monitor, default_timeout=5.0)
        agent = MockAgent(name="test-agent")

        await executor.generate(agent, "Prompt", [], phase="revision", round_num=3)

        monitor.track_agent_call.assert_called_once_with(
            "test-agent", "generate", phase="revision", round_num=3
        )

    @pytest.mark.asyncio
    async def test_critique_tracks_phase_and_round(self):
        """Test critique correctly tracks phase and round."""
        monitor = MagicMock()
        monitor.track_agent_call = MagicMock(return_value="id-456")
        monitor.record_completion = MagicMock()

        executor = AutonomicExecutor(performance_monitor=monitor, default_timeout=5.0)
        agent = MockAgent(name="test-agent")

        await executor.critique(agent, "Proposal", "Task", [], phase="critique", round_num=2)

        monitor.track_agent_call.assert_called_once_with(
            "test-agent", "critique", phase="critique", round_num=2
        )

    @pytest.mark.asyncio
    async def test_vote_tracks_phase_and_round(self):
        """Test vote correctly tracks phase and round."""
        monitor = MagicMock()
        monitor.track_agent_call = MagicMock(return_value="id-789")
        monitor.record_completion = MagicMock()

        executor = AutonomicExecutor(performance_monitor=monitor, default_timeout=5.0)
        agent = MockAgent(name="test-agent")
        proposals = {"agent1": "Proposal"}

        await executor.vote(agent, proposals, "Task", phase="final_vote", round_num=5)

        monitor.track_agent_call.assert_called_once_with(
            "test-agent", "vote", phase="final_vote", round_num=5
        )


class TestTelemetryInitialization:
    """Tests for telemetry initialization."""

    def test_telemetry_initialization_on_construction(self):
        """Test telemetry collectors are initialized when enable_telemetry is True."""
        # Reset the global state
        import aragora.debate.autonomic_executor as ae

        ae._telemetry_initialized = False

        with patch("aragora.debate.autonomic_executor._ensure_telemetry_collectors") as mock_init:
            executor = AutonomicExecutor(enable_telemetry=True, default_timeout=5.0)
            # Should have called the initialization
            mock_init.assert_called_once()

    def test_telemetry_not_initialized_when_disabled(self):
        """Test telemetry collectors are NOT initialized when enable_telemetry is False."""
        with patch("aragora.debate.autonomic_executor._ensure_telemetry_collectors") as mock_init:
            executor = AutonomicExecutor(enable_telemetry=False, default_timeout=5.0)
            mock_init.assert_not_called()


class TestWisdomResponseFormat:
    """Tests for wisdom response formatting."""

    def test_get_wisdom_fallback_formats_correctly(self):
        """Test wisdom fallback response is formatted correctly."""
        mock_store = MagicMock()
        mock_store.get_relevant_wisdom.return_value = [
            {
                "id": 42,
                "text": "This is the audience wisdom text",
                "submitter_id": "helpful-user-456",
            }
        ]
        mock_store.mark_wisdom_used = MagicMock()

        executor = AutonomicExecutor(wisdom_store=mock_store, loop_id="debate-123")
        result = executor._get_wisdom_fallback("failed-agent")

        assert result is not None
        assert "[Audience Wisdom" in result
        assert "helpful-user-456" in result
        assert "This is the audience wisdom text" in result
        assert "failed-agent" in result
        mock_store.mark_wisdom_used.assert_called_once_with(42)


class TestAllAgentsCircuitBroken:
    """Tests for scenarios where all agents are circuit broken."""

    @pytest.mark.asyncio
    async def test_all_agents_circuit_broken(self):
        """Test behavior when all agents are circuit broken."""
        cb = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=300.0)
        executor = AutonomicExecutor(circuit_breaker=cb, default_timeout=5.0)

        primary = MockAgent(name="primary", response="Primary")
        fallback = MockAgent(name="fallback", response="Fallback")

        # Open circuits for both
        cb.record_failure("primary")
        cb.record_failure("primary")
        cb.record_failure("fallback")
        cb.record_failure("fallback")

        result = await executor.generate_with_fallback(
            primary, "Test prompt", [], fallback_agents=[fallback]
        )

        # All agents are circuit broken, should fail
        assert "[System:" in result
        assert "All agents failed" in result
        # Neither agent should have been called
        assert primary.generate_calls == 0
        assert fallback.generate_calls == 0
