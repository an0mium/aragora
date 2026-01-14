"""
Tests for the AutonomicExecutor safe operation handler.

Tests cover:
- StreamingContentBuffer async buffer operations
- AutonomicExecutor timeout escalation and retry behavior
- Error handling for generate, critique, and vote operations
- Fallback agent substitution
- Circuit breaker integration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mark tests as slow (involve timeout escalation scenarios)
pytestmark = pytest.mark.slow

from aragora.core import Agent, Critique, Message, Vote
from aragora.debate.autonomic_executor import (
    AutonomicExecutor,
    StreamingContentBuffer,
)
from aragora.resilience import CircuitBreaker


# === Fixtures ===


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent", model: str = "test_model"):
        super().__init__(name, model)
        self.generate_response = "test response"
        self.generate_delay = 0.0
        self.should_raise = False
        self.raise_error = Exception("test error")
        self.call_count = 0

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        self.call_count += 1
        if self.generate_delay > 0:
            await asyncio.sleep(self.generate_delay)
        if self.should_raise:
            raise self.raise_error
        return self.generate_response

    async def critique(
        self, proposal: str, task: str, context: list[Message] | None = None
    ) -> Critique:
        self.call_count += 1
        if self.generate_delay > 0:
            await asyncio.sleep(self.generate_delay)
        if self.should_raise:
            raise self.raise_error
        return Critique(
            agent=self.name,
            target_agent="other_agent",
            target_content=proposal[:100],
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        self.call_count += 1
        if self.generate_delay > 0:
            await asyncio.sleep(self.generate_delay)
        if self.should_raise:
            raise self.raise_error
        return Vote(
            agent=self.name,
            choice=list(proposals.keys())[0] if proposals else "unknown",
            reasoning="Test vote reasoning",
            confidence=0.8,
        )


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent()


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker for testing."""
    return CircuitBreaker(failure_threshold=3, cooldown_seconds=1.0)


@pytest.fixture
def executor(circuit_breaker):
    """Create an AutonomicExecutor with default settings."""
    return AutonomicExecutor(
        circuit_breaker=circuit_breaker,
        default_timeout=1.0,
        timeout_escalation_factor=1.5,
        max_timeout=10.0,
    )


@pytest.fixture
def streaming_buffer():
    """Create a StreamingContentBuffer for testing."""
    return StreamingContentBuffer()


# === StreamingContentBuffer Tests ===


class TestStreamingContentBuffer:
    """Tests for StreamingContentBuffer."""

    async def test_append_and_get(self, streaming_buffer):
        """Test basic append and get operations."""
        await streaming_buffer.append("agent1", "chunk1")
        await streaming_buffer.append("agent1", "chunk2")

        result = streaming_buffer.get_partial("agent1")
        assert result == "chunk1chunk2"

    async def test_get_partial_empty(self, streaming_buffer):
        """Test getting partial for unknown agent."""
        result = streaming_buffer.get_partial("unknown")
        assert result == ""

    async def test_get_partial_async(self, streaming_buffer):
        """Test async get with lock."""
        await streaming_buffer.append("agent1", "content")
        result = await streaming_buffer.get_partial_async("agent1")
        assert result == "content"

    async def test_clear(self, streaming_buffer):
        """Test clearing buffer."""
        await streaming_buffer.append("agent1", "content")
        await streaming_buffer.clear("agent1")
        assert streaming_buffer.get_partial("agent1") == ""

    def test_clear_sync(self, streaming_buffer):
        """Test synchronous clear."""
        # Manually set buffer for sync test
        streaming_buffer._buffer["agent1"] = "content"
        streaming_buffer.clear_sync("agent1")
        assert streaming_buffer.get_partial("agent1") == ""

    async def test_separate_agents(self, streaming_buffer):
        """Test that different agents have separate buffers."""
        await streaming_buffer.append("agent1", "content1")
        await streaming_buffer.append("agent2", "content2")

        assert streaming_buffer.get_partial("agent1") == "content1"
        assert streaming_buffer.get_partial("agent2") == "content2"

    async def test_concurrent_access(self, streaming_buffer):
        """Test concurrent access to same agent's buffer."""

        async def append_chunks(agent: str, prefix: str, count: int):
            for i in range(count):
                await streaming_buffer.append(agent, f"{prefix}{i}")
                await asyncio.sleep(0.001)

        await asyncio.gather(
            append_chunks("agent1", "a", 10),
            append_chunks("agent1", "b", 10),
        )

        result = streaming_buffer.get_partial("agent1")
        # Should have all chunks from both tasks
        assert len(result) == 20 * 2  # 10 chunks each of length 2

    async def test_clear_unknown_agent(self, streaming_buffer):
        """Test clearing non-existent agent (should not raise)."""
        await streaming_buffer.clear("unknown")
        streaming_buffer.clear_sync("unknown")
        # Should not raise


# === AutonomicExecutor Basic Tests ===


class TestAutonomicExecutorBasic:
    """Basic tests for AutonomicExecutor initialization and properties."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        executor = AutonomicExecutor()
        assert executor.circuit_breaker is None
        assert executor.default_timeout > 0
        assert executor.timeout_escalation_factor == 1.5
        assert executor.max_timeout == 600.0

    def test_init_with_custom_values(self, circuit_breaker):
        """Test initialization with custom values."""
        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=30.0,
            timeout_escalation_factor=2.0,
            max_timeout=120.0,
        )
        assert executor.circuit_breaker is circuit_breaker
        assert executor.default_timeout == 30.0
        assert executor.timeout_escalation_factor == 2.0
        assert executor.max_timeout == 120.0

    def test_set_loop_id(self, executor):
        """Test setting loop ID."""
        executor.set_loop_id("test-loop-123")
        assert executor.loop_id == "test-loop-123"


# === Timeout Escalation Tests ===


class TestTimeoutEscalation:
    """Tests for timeout escalation behavior."""

    def test_get_escalated_timeout_initial(self, executor):
        """Test initial timeout (no retries)."""
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 1.0  # Default timeout

    def test_get_escalated_timeout_after_retry(self, executor):
        """Test timeout increases after retry."""
        executor.record_retry("agent1")
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 1.5  # 1.0 * 1.5

    def test_get_escalated_timeout_multiple_retries(self, executor):
        """Test timeout escalates exponentially."""
        executor.record_retry("agent1")
        executor.record_retry("agent1")
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 2.25  # 1.0 * 1.5 * 1.5

    def test_get_escalated_timeout_capped(self, executor):
        """Test timeout is capped at max_timeout."""
        for _ in range(20):
            executor.record_retry("agent1")
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 10.0  # max_timeout

    def test_get_escalated_timeout_with_base(self, executor):
        """Test custom base timeout."""
        timeout = executor.get_escalated_timeout("agent1", base_timeout=5.0)
        assert timeout == 5.0

    def test_record_retry_returns_count(self, executor):
        """Test record_retry returns new count."""
        count1 = executor.record_retry("agent1")
        count2 = executor.record_retry("agent1")
        assert count1 == 1
        assert count2 == 2

    def test_reset_retries(self, executor):
        """Test resetting retry count."""
        executor.record_retry("agent1")
        executor.record_retry("agent1")
        executor.reset_retries("agent1")
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 1.0  # Back to default

    def test_reset_retries_unknown_agent(self, executor):
        """Test resetting unknown agent (should not raise)."""
        executor.reset_retries("unknown")


# === with_timeout Tests ===


class TestWithTimeout:
    """Tests for with_timeout method."""

    async def test_with_timeout_success(self, executor):
        """Test successful execution within timeout."""

        async def fast_coro():
            return "success"

        result = await executor.with_timeout(fast_coro(), "agent1", 1.0)
        assert result == "success"

    async def test_with_timeout_raises_on_timeout(self, executor):
        """Test TimeoutError is raised on timeout."""

        async def slow_coro():
            await asyncio.sleep(2.0)
            return "too late"

        with pytest.raises(TimeoutError, match="Agent agent1 timed out"):
            await executor.with_timeout(slow_coro(), "agent1", 0.1)

    async def test_with_timeout_records_circuit_breaker_failure(self, executor, circuit_breaker):
        """Test that timeout records circuit breaker failure."""

        async def slow_coro():
            await asyncio.sleep(2.0)

        try:
            await executor.with_timeout(slow_coro(), "agent1", 0.1)
        except TimeoutError:
            pass

        # Should have recorded failure - check using _failures dict
        assert circuit_breaker._failures.get("agent1", 0) >= 1


# === Generate Tests ===


class TestAutonomicExecutorGenerate:
    """Tests for AutonomicExecutor.generate() method."""

    async def test_generate_success(self, executor, mock_agent):
        """Test successful generation."""
        result = await executor.generate(mock_agent, "prompt", [])
        assert result == "test response"

    async def test_generate_timeout_returns_system_message(self, executor, mock_agent):
        """Test timeout returns system message."""
        mock_agent.should_raise = True
        mock_agent.raise_error = asyncio.TimeoutError("Simulated timeout")
        result = await executor.generate(mock_agent, "prompt", [])
        assert "[System:" in result
        assert "timed out" in result

    async def test_generate_error_returns_system_message(self, executor, mock_agent):
        """Test error returns system message."""
        mock_agent.should_raise = True
        result = await executor.generate(mock_agent, "prompt", [])
        assert "[System:" in result
        assert "error" in result.lower()

    async def test_generate_connection_error_returns_system_message(self, executor, mock_agent):
        """Test connection error returns system message."""
        mock_agent.should_raise = True
        mock_agent.raise_error = ConnectionError("Connection refused")
        result = await executor.generate(mock_agent, "prompt", [])
        assert "[System:" in result
        assert "connection failed" in result.lower()

    async def test_generate_sanitizes_output(self, executor, mock_agent):
        """Test that output is sanitized."""
        mock_agent.generate_response = "  \n  test response  \n  "
        result = await executor.generate(mock_agent, "prompt", [])
        # Sanitizer should clean up whitespace
        assert result.strip() == result or "test response" in result


# === Critique Tests ===


class TestAutonomicExecutorCritique:
    """Tests for AutonomicExecutor.critique() method."""

    async def test_critique_success(self, executor, mock_agent):
        """Test successful critique."""
        result = await executor.critique(mock_agent, "proposal", "task", [])
        assert isinstance(result, Critique)
        assert result.agent == "test_agent"

    async def test_critique_timeout_returns_none(self, executor, mock_agent):
        """Test timeout returns None."""
        mock_agent.should_raise = True
        mock_agent.raise_error = asyncio.TimeoutError("Simulated timeout")
        result = await executor.critique(mock_agent, "proposal", "task", [])
        assert result is None

    async def test_critique_error_returns_none(self, executor, mock_agent):
        """Test error returns None."""
        mock_agent.should_raise = True
        result = await executor.critique(mock_agent, "proposal", "task", [])
        assert result is None

    async def test_critique_connection_error_returns_none(self, executor, mock_agent):
        """Test connection error returns None."""
        mock_agent.should_raise = True
        mock_agent.raise_error = OSError("Network unreachable")
        result = await executor.critique(mock_agent, "proposal", "task", [])
        assert result is None


# === Vote Tests ===


class TestAutonomicExecutorVote:
    """Tests for AutonomicExecutor.vote() method."""

    async def test_vote_success(self, executor, mock_agent):
        """Test successful vote."""
        proposals = {"agent_a": "proposal a", "agent_b": "proposal b"}
        result = await executor.vote(mock_agent, proposals, "task")
        assert isinstance(result, Vote)
        assert result.agent == "test_agent"

    async def test_vote_timeout_returns_none(self, executor, mock_agent):
        """Test timeout returns None."""
        mock_agent.should_raise = True
        mock_agent.raise_error = asyncio.TimeoutError("Simulated timeout")
        result = await executor.vote(mock_agent, {"a": "prop"}, "task")
        assert result is None

    async def test_vote_error_returns_none(self, executor, mock_agent):
        """Test error returns None."""
        mock_agent.should_raise = True
        result = await executor.vote(mock_agent, {"a": "prop"}, "task")
        assert result is None


# === Generate with Fallback Tests ===


class TestGenerateWithFallback:
    """Tests for generate_with_fallback method."""

    async def test_fallback_primary_success(self, executor, mock_agent):
        """Test primary agent succeeds, no fallback needed."""
        result = await executor.generate_with_fallback(mock_agent, "prompt", [], fallback_agents=[])
        assert result == "test response"

    async def test_fallback_uses_backup_agent(self, executor):
        """Test fallback to backup agent when primary fails."""
        primary = MockAgent("primary")
        primary.should_raise = True

        backup = MockAgent("backup")
        backup.generate_response = "backup response"

        result = await executor.generate_with_fallback(
            primary, "prompt", [], fallback_agents=[backup], max_retries=1
        )
        assert result == "backup response"

    async def test_fallback_all_agents_fail(self, executor):
        """Test when all agents fail."""
        primary = MockAgent("primary")
        primary.should_raise = True

        backup = MockAgent("backup")
        backup.should_raise = True

        result = await executor.generate_with_fallback(
            primary, "prompt", [], fallback_agents=[backup], max_retries=1
        )
        assert "[System: All agents failed" in result
        assert "primary" in result
        assert "backup" in result

    async def test_fallback_skips_circuit_broken_agent(self, executor, circuit_breaker):
        """Test that circuit-broken agents are skipped."""
        primary = MockAgent("primary")
        backup = MockAgent("backup")
        backup.generate_response = "backup response"

        # Mark primary as circuit-broken
        for _ in range(5):
            circuit_breaker.record_failure("primary")

        result = await executor.generate_with_fallback(
            primary, "prompt", [], fallback_agents=[backup], max_retries=1
        )
        assert result == "backup response"
        assert primary.call_count == 0  # Primary was skipped

    async def test_fallback_retries_before_moving_on(self, executor):
        """Test that agent is retried before moving to fallback."""
        primary = MockAgent("primary")
        primary.generate_delay = 2.0  # Will timeout

        backup = MockAgent("backup")
        backup.generate_response = "backup response"

        result = await executor.generate_with_fallback(
            primary, "prompt", [], fallback_agents=[backup], max_retries=2
        )
        assert result == "backup response"
        # Primary should have been tried twice (max_retries=2)
        assert primary.call_count == 2

    async def test_fallback_uses_partial_content(self, circuit_breaker):
        """Test that partial content is used when available during streaming timeout.

        Note: The streaming buffer is cleared before each attempt, so partial content
        must be accumulated during the actual generate call. This test simulates that
        by having the mock agent append to the buffer while generating.
        """
        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=0.1,  # Short timeout
        )

        class StreamingMockAgent(MockAgent):
            async def generate(self, prompt, context=None):
                # Simulate streaming by appending to buffer before timing out
                partial_text = "Partial content from streaming that is long enough " * 5
                await executor.streaming_buffer.append(self.name, partial_text)
                # Then timeout
                await asyncio.sleep(2.0)
                return "never reached"

        primary = StreamingMockAgent("primary")

        result = await executor.generate_with_fallback(
            primary, "prompt", [], fallback_agents=[], max_retries=1
        )
        assert "Partial content" in result
        assert "[System: Response truncated" in result


# === Wisdom Fallback Tests ===


class TestWisdomFallback:
    """Tests for audience wisdom fallback."""

    def test_get_wisdom_fallback_no_store(self, executor):
        """Test wisdom fallback returns None without store."""
        result = executor._get_wisdom_fallback("agent1")
        assert result is None

    def test_get_wisdom_fallback_no_loop_id(self, executor):
        """Test wisdom fallback returns None without loop_id."""
        executor.wisdom_store = MagicMock()
        result = executor._get_wisdom_fallback("agent1")
        assert result is None

    def test_get_wisdom_fallback_returns_wisdom(self, executor):
        """Test wisdom fallback returns formatted wisdom."""
        mock_store = MagicMock()
        mock_store.get_relevant_wisdom.return_value = [
            {"id": "w1", "text": "Audience insight", "submitter_id": "user123"}
        ]

        executor.wisdom_store = mock_store
        executor.loop_id = "loop123"

        result = executor._get_wisdom_fallback("failed_agent")
        assert "[Audience Wisdom" in result
        assert "Audience insight" in result
        assert "user123" in result
        mock_store.mark_wisdom_used.assert_called_once_with("w1")


# === Telemetry Tests ===


class TestTelemetry:
    """Tests for telemetry emission."""

    def test_telemetry_disabled_by_default(self, executor, mock_agent):
        """Test telemetry is disabled by default."""
        assert executor.enable_telemetry is False

    async def test_telemetry_not_emitted_when_disabled(self, executor, mock_agent):
        """Test telemetry methods are not called when disabled."""
        # When telemetry is disabled, the _emit_agent_telemetry method should return early
        # We can verify by checking that generate still works without telemetry setup
        assert executor.enable_telemetry is False
        result = await executor.generate(mock_agent, "prompt", [])
        assert result == "test response"  # Should work without telemetry

    async def test_telemetry_enabled_init(self, circuit_breaker):
        """Test telemetry can be enabled on init."""
        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            enable_telemetry=True,
        )
        assert executor.enable_telemetry is True


# === Performance Monitor Integration Tests ===


class TestPerformanceMonitorIntegration:
    """Tests for performance monitor integration."""

    async def test_performance_monitor_tracks_generate(self, circuit_breaker, mock_agent):
        """Test that performance monitor is called for generate."""
        mock_monitor = MagicMock()
        mock_monitor.track_agent_call.return_value = "tracking-id-123"

        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=1.0,
            performance_monitor=mock_monitor,
        )

        await executor.generate(mock_agent, "prompt", [], phase="proposal", round_num=1)

        mock_monitor.track_agent_call.assert_called_once_with(
            "test_agent", "generate", phase="proposal", round_num=1
        )
        mock_monitor.record_completion.assert_called_once()

    async def test_performance_monitor_tracks_failure(self, circuit_breaker, mock_agent):
        """Test that performance monitor tracks failures."""
        mock_agent.should_raise = True

        mock_monitor = MagicMock()
        mock_monitor.track_agent_call.return_value = "tracking-id-123"

        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=1.0,
            performance_monitor=mock_monitor,
        )

        await executor.generate(mock_agent, "prompt", [])

        # Should have recorded completion with success=False
        call_args = mock_monitor.record_completion.call_args
        assert call_args[1]["success"] is False


# === Immune System Integration Tests ===


class TestImmuneSystemIntegration:
    """Tests for immune system integration."""

    async def test_immune_system_notified_on_success(self, circuit_breaker, mock_agent):
        """Test that immune system is notified on success."""
        mock_immune = MagicMock()

        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=1.0,
            immune_system=mock_immune,
        )

        await executor.generate(mock_agent, "prompt", [])

        mock_immune.agent_started.assert_called_once()
        mock_immune.agent_completed.assert_called_once()

    async def test_immune_system_notified_on_timeout(self, circuit_breaker, mock_agent):
        """Test that immune system is notified on timeout."""
        mock_agent.should_raise = True
        mock_agent.raise_error = asyncio.TimeoutError("Simulated timeout")
        mock_immune = MagicMock()

        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=1.0,
            immune_system=mock_immune,
        )

        await executor.generate(mock_agent, "prompt", [])

        mock_immune.agent_started.assert_called_once()
        mock_immune.agent_timeout.assert_called_once()

    async def test_immune_system_notified_on_failure(self, circuit_breaker, mock_agent):
        """Test that immune system is notified on failure."""
        mock_agent.should_raise = True
        mock_immune = MagicMock()

        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=1.0,
            immune_system=mock_immune,
        )

        await executor.generate(mock_agent, "prompt", [])

        mock_immune.agent_started.assert_called_once()
        mock_immune.agent_failed.assert_called_once()


# === Chaos Director Integration Tests ===


class TestChaosDirectorIntegration:
    """Tests for chaos director integration (theatrical failure messages)."""

    async def test_chaos_director_timeout_message(self, circuit_breaker, mock_agent):
        """Test that chaos director provides timeout message."""
        mock_agent.should_raise = True
        mock_agent.raise_error = asyncio.TimeoutError("Simulated timeout")

        mock_chaos = MagicMock()
        mock_chaos.timeout_response.return_value.message = (
            "[THEATRICAL] Agent was vanquished by time!"
        )

        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=1.0,
            chaos_director=mock_chaos,
        )

        result = await executor.generate(mock_agent, "prompt", [])
        assert "[THEATRICAL]" in result
        mock_chaos.timeout_response.assert_called_once()

    async def test_chaos_director_connection_message(self, circuit_breaker, mock_agent):
        """Test that chaos director provides connection error message."""
        mock_agent.should_raise = True
        mock_agent.raise_error = ConnectionError("Network down")

        mock_chaos = MagicMock()
        mock_chaos.connection_response.return_value.message = (
            "[THEATRICAL] The network spirits have abandoned us!"
        )

        executor = AutonomicExecutor(
            circuit_breaker=circuit_breaker,
            default_timeout=1.0,
            chaos_director=mock_chaos,
        )

        result = await executor.generate(mock_agent, "prompt", [])
        assert "[THEATRICAL]" in result
        mock_chaos.connection_response.assert_called_once()
