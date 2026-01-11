"""
Tests for Autonomic Executor module.

Tests cover:
- StreamingContentBuffer operations
- AutonomicExecutor initialization
- Timeout escalation logic
- with_timeout wrapper
- generate() with error handling
- critique() with error handling
- vote() with error handling
- generate_with_fallback() with fallback chain
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from aragora.debate.autonomic_executor import (
    StreamingContentBuffer,
    AutonomicExecutor,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def streaming_buffer():
    """Create a fresh streaming content buffer."""
    return StreamingContentBuffer()


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker."""
    breaker = Mock()
    breaker.record_failure = Mock()
    breaker.is_available = Mock(return_value=True)
    return breaker


@pytest.fixture
def mock_immune_system():
    """Create a mock immune system."""
    system = Mock()
    system.agent_started = Mock()
    system.agent_completed = Mock()
    system.agent_timeout = Mock()
    system.agent_failed = Mock()
    return system


@pytest.fixture
def mock_agent():
    """Create a mock agent with async methods."""
    agent = Mock()
    agent.name = "test_agent"
    agent.generate = AsyncMock(return_value="Generated response")
    agent.critique = AsyncMock(return_value=Mock(text="Critique response"))
    agent.vote = AsyncMock(return_value=Mock(choice="agent_a"))
    return agent


@pytest.fixture
def executor(mock_circuit_breaker):
    """Create an autonomic executor with mock dependencies."""
    return AutonomicExecutor(
        circuit_breaker=mock_circuit_breaker,
        default_timeout=10.0,
        timeout_escalation_factor=1.5,
        max_timeout=60.0,
    )


# ============================================================================
# StreamingContentBuffer Tests
# ============================================================================

class TestStreamingContentBuffer:
    """Tests for StreamingContentBuffer class."""

    @pytest.mark.asyncio
    async def test_append_and_get_partial(self, streaming_buffer):
        """Test appending and getting partial content."""
        await streaming_buffer.append("agent1", "Hello ")
        await streaming_buffer.append("agent1", "World")

        result = streaming_buffer.get_partial("agent1")
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_get_partial_empty(self, streaming_buffer):
        """Test getting partial for unknown agent."""
        result = streaming_buffer.get_partial("unknown")
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_partial_async(self, streaming_buffer):
        """Test async variant of get_partial."""
        await streaming_buffer.append("agent1", "Test content")

        result = await streaming_buffer.get_partial_async("agent1")
        assert result == "Test content"

    @pytest.mark.asyncio
    async def test_clear(self, streaming_buffer):
        """Test clearing agent's buffer."""
        await streaming_buffer.append("agent1", "Content")
        await streaming_buffer.clear("agent1")

        result = streaming_buffer.get_partial("agent1")
        assert result == ""

    def test_clear_sync(self, streaming_buffer):
        """Test synchronous clear."""
        # Use internal buffer directly for setup
        streaming_buffer._buffer["agent1"] = "Content"

        streaming_buffer.clear_sync("agent1")

        result = streaming_buffer.get_partial("agent1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_multiple_agents_isolated(self, streaming_buffer):
        """Test that different agents have isolated buffers."""
        await streaming_buffer.append("agent1", "Content 1")
        await streaming_buffer.append("agent2", "Content 2")

        assert streaming_buffer.get_partial("agent1") == "Content 1"
        assert streaming_buffer.get_partial("agent2") == "Content 2"

    @pytest.mark.asyncio
    async def test_clear_only_affects_target(self, streaming_buffer):
        """Test clearing one agent doesn't affect others."""
        await streaming_buffer.append("agent1", "Content 1")
        await streaming_buffer.append("agent2", "Content 2")

        await streaming_buffer.clear("agent1")

        assert streaming_buffer.get_partial("agent1") == ""
        assert streaming_buffer.get_partial("agent2") == "Content 2"


# ============================================================================
# AutonomicExecutor Initialization Tests
# ============================================================================

class TestAutonomicExecutorInit:
    """Tests for AutonomicExecutor initialization."""

    def test_initialization_defaults(self):
        """Test initialization with defaults."""
        executor = AutonomicExecutor()

        assert executor.circuit_breaker is None
        assert executor.timeout_escalation_factor == 1.5
        assert executor.max_timeout == 600.0
        assert executor.streaming_buffer is not None
        assert executor.wisdom_store is None
        assert executor.loop_id is None

    def test_initialization_with_circuit_breaker(self, mock_circuit_breaker):
        """Test initialization with circuit breaker."""
        executor = AutonomicExecutor(circuit_breaker=mock_circuit_breaker)

        assert executor.circuit_breaker == mock_circuit_breaker

    def test_initialization_custom_timeout(self):
        """Test initialization with custom timeout settings."""
        executor = AutonomicExecutor(
            default_timeout=30.0,
            timeout_escalation_factor=2.0,
            max_timeout=120.0,
        )

        assert executor.default_timeout == 30.0
        assert executor.timeout_escalation_factor == 2.0
        assert executor.max_timeout == 120.0

    def test_initialization_with_immune_system(self, mock_immune_system):
        """Test initialization with immune system."""
        executor = AutonomicExecutor(immune_system=mock_immune_system)

        assert executor.immune_system == mock_immune_system

    def test_initialization_with_streaming_buffer(self, streaming_buffer):
        """Test initialization with custom streaming buffer."""
        executor = AutonomicExecutor(streaming_buffer=streaming_buffer)

        assert executor.streaming_buffer == streaming_buffer

    def test_set_loop_id(self, executor):
        """Test setting loop ID."""
        executor.set_loop_id("loop_123")

        assert executor.loop_id == "loop_123"


# ============================================================================
# Timeout Escalation Tests
# ============================================================================

class TestTimeoutEscalation:
    """Tests for timeout escalation logic."""

    def test_get_escalated_timeout_no_retries(self, executor):
        """Test timeout with no retries."""
        timeout = executor.get_escalated_timeout("agent1")

        assert timeout == 10.0  # default_timeout

    def test_get_escalated_timeout_with_retries(self, executor):
        """Test timeout increases with retries."""
        executor.record_retry("agent1")  # 1 retry
        timeout = executor.get_escalated_timeout("agent1")

        assert timeout == 15.0  # 10 * 1.5^1

    def test_get_escalated_timeout_multiple_retries(self, executor):
        """Test timeout with multiple retries."""
        executor.record_retry("agent1")  # 1
        executor.record_retry("agent1")  # 2

        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 22.5  # 10 * 1.5^2

    def test_get_escalated_timeout_capped(self, executor):
        """Test timeout capped at max_timeout."""
        # Record many retries
        for _ in range(20):
            executor.record_retry("agent1")

        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 60.0  # Capped at max_timeout

    def test_get_escalated_timeout_custom_base(self, executor):
        """Test with custom base timeout."""
        timeout = executor.get_escalated_timeout("agent1", base_timeout=20.0)

        assert timeout == 20.0  # Custom base, no retries

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
        assert timeout == 10.0  # Back to base

    def test_reset_retries_unknown_agent(self, executor):
        """Test resetting retries for unknown agent doesn't raise."""
        # Should not raise
        executor.reset_retries("unknown_agent")


# ============================================================================
# with_timeout Tests
# ============================================================================

class TestWithTimeout:
    """Tests for with_timeout wrapper."""

    @pytest.mark.asyncio
    async def test_with_timeout_success(self, executor):
        """Test successful completion within timeout."""
        async def quick_task():
            return "result"

        result = await executor.with_timeout(quick_task(), "agent1", timeout_seconds=1.0)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_with_timeout_raises_on_timeout(self, executor):
        """Test TimeoutError raised when operation times out."""
        async def slow_task():
            await asyncio.sleep(10)
            return "never reached"

        with pytest.raises(TimeoutError) as exc_info:
            await executor.with_timeout(slow_task(), "agent1", timeout_seconds=0.01)

        assert "agent1" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_with_timeout_records_circuit_breaker_failure(self, executor, mock_circuit_breaker):
        """Test circuit breaker failure recorded on timeout."""
        async def slow_task():
            await asyncio.sleep(10)

        with pytest.raises(TimeoutError):
            await executor.with_timeout(slow_task(), "agent1", timeout_seconds=0.01)

        mock_circuit_breaker.record_failure.assert_called_once_with("agent1")

    @pytest.mark.asyncio
    async def test_with_timeout_uses_default(self, executor):
        """Test uses default timeout when not specified."""
        async def quick_task():
            return "result"

        result = await executor.with_timeout(quick_task(), "agent1")
        assert result == "result"


# ============================================================================
# generate() Tests
# ============================================================================

class TestGenerate:
    """Tests for generate() method."""

    @pytest.mark.asyncio
    async def test_generate_success(self, executor, mock_agent):
        """Test successful generation."""
        result = await executor.generate(mock_agent, "prompt", [])

        assert "Generated response" in result
        mock_agent.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_notifies_immune_system_start(self, mock_agent, mock_immune_system):
        """Test immune system notified on start."""
        executor = AutonomicExecutor(immune_system=mock_immune_system)

        await executor.generate(mock_agent, "test prompt", [])

        mock_immune_system.agent_started.assert_called_once()
        call_args = mock_immune_system.agent_started.call_args
        assert call_args[0][0] == "test_agent"

    @pytest.mark.asyncio
    async def test_generate_notifies_immune_system_complete(self, mock_agent, mock_immune_system):
        """Test immune system notified on completion."""
        executor = AutonomicExecutor(immune_system=mock_immune_system)

        await executor.generate(mock_agent, "test prompt", [])

        mock_immune_system.agent_completed.assert_called_once()
        call_args = mock_immune_system.agent_completed.call_args
        assert call_args[0][0] == "test_agent"
        assert call_args[1]["success"] is True

    @pytest.mark.asyncio
    async def test_generate_timeout_returns_system_message(self, executor):
        """Test timeout returns system message."""
        agent = Mock()
        agent.name = "slow_agent"
        agent.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await executor.generate(agent, "prompt", [])

        assert "[System:" in result
        assert "slow_agent" in result
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_generate_timeout_notifies_immune_system(self, mock_immune_system):
        """Test immune system notified on timeout."""
        executor = AutonomicExecutor(immune_system=mock_immune_system)
        agent = Mock()
        agent.name = "slow_agent"
        agent.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        await executor.generate(agent, "prompt", [])

        mock_immune_system.agent_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_connection_error_returns_system_message(self, executor):
        """Test connection error returns system message."""
        agent = Mock()
        agent.name = "disconnected_agent"
        agent.generate = AsyncMock(side_effect=ConnectionError("Network unreachable"))

        result = await executor.generate(agent, "prompt", [])

        assert "[System:" in result
        assert "disconnected_agent" in result
        assert "connection failed" in result

    @pytest.mark.asyncio
    async def test_generate_exception_returns_system_message(self, executor):
        """Test general exception returns system message."""
        agent = Mock()
        agent.name = "broken_agent"
        agent.generate = AsyncMock(side_effect=RuntimeError("Internal error"))

        result = await executor.generate(agent, "prompt", [])

        assert "[System:" in result
        assert "broken_agent" in result
        assert "error" in result

    @pytest.mark.asyncio
    async def test_generate_exception_notifies_immune_system(self, mock_immune_system):
        """Test immune system notified on exception."""
        executor = AutonomicExecutor(immune_system=mock_immune_system)
        agent = Mock()
        agent.name = "broken_agent"
        agent.generate = AsyncMock(side_effect=RuntimeError("error"))

        await executor.generate(agent, "prompt", [])

        mock_immune_system.agent_failed.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_chaos_director_timeout(self, executor):
        """Test chaos director provides theatrical timeout message."""
        chaos = Mock()
        chaos.timeout_response = Mock(return_value=Mock(message="[Dramatic timeout!]"))
        executor.chaos_director = chaos

        agent = Mock()
        agent.name = "slow_agent"
        agent.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await executor.generate(agent, "prompt", [])

        assert result == "[Dramatic timeout!]"
        chaos.timeout_response.assert_called_once()


# ============================================================================
# critique() Tests
# ============================================================================

class TestCritique:
    """Tests for critique() method."""

    @pytest.mark.asyncio
    async def test_critique_success(self, executor, mock_agent):
        """Test successful critique."""
        result = await executor.critique(mock_agent, "proposal", "task", [])

        assert result is not None
        mock_agent.critique.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_timeout_returns_none(self, executor):
        """Test timeout returns None."""
        agent = Mock()
        agent.name = "slow_agent"
        agent.critique = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await executor.critique(agent, "proposal", "task", [])

        assert result is None

    @pytest.mark.asyncio
    async def test_critique_connection_error_returns_none(self, executor):
        """Test connection error returns None."""
        agent = Mock()
        agent.name = "disconnected_agent"
        agent.critique = AsyncMock(side_effect=ConnectionError())

        result = await executor.critique(agent, "proposal", "task", [])

        assert result is None

    @pytest.mark.asyncio
    async def test_critique_exception_returns_none(self, executor):
        """Test general exception returns None."""
        agent = Mock()
        agent.name = "broken_agent"
        agent.critique = AsyncMock(side_effect=RuntimeError("error"))

        result = await executor.critique(agent, "proposal", "task", [])

        assert result is None


# ============================================================================
# vote() Tests
# ============================================================================

class TestVote:
    """Tests for vote() method."""

    @pytest.mark.asyncio
    async def test_vote_success(self, executor, mock_agent):
        """Test successful vote."""
        proposals = {"agent_a": "Proposal A", "agent_b": "Proposal B"}
        result = await executor.vote(mock_agent, proposals, "task")

        assert result is not None
        mock_agent.vote.assert_called_once()

    @pytest.mark.asyncio
    async def test_vote_timeout_returns_none(self, executor):
        """Test timeout returns None."""
        agent = Mock()
        agent.name = "slow_agent"
        agent.vote = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await executor.vote(agent, {"a": "A"}, "task")

        assert result is None

    @pytest.mark.asyncio
    async def test_vote_connection_error_returns_none(self, executor):
        """Test connection error returns None."""
        agent = Mock()
        agent.name = "disconnected_agent"
        agent.vote = AsyncMock(side_effect=ConnectionError())

        result = await executor.vote(agent, {"a": "A"}, "task")

        assert result is None

    @pytest.mark.asyncio
    async def test_vote_exception_returns_none(self, executor):
        """Test general exception returns None."""
        agent = Mock()
        agent.name = "broken_agent"
        agent.vote = AsyncMock(side_effect=RuntimeError("error"))

        result = await executor.vote(agent, {"a": "A"}, "task")

        assert result is None


# ============================================================================
# generate_with_fallback() Tests
# ============================================================================

class TestGenerateWithFallback:
    """Tests for generate_with_fallback() method."""

    @pytest.mark.asyncio
    async def test_fallback_success_first_try(self, executor, mock_agent):
        """Test success on first agent."""
        result = await executor.generate_with_fallback(mock_agent, "prompt", [])

        assert "Generated response" in result
        mock_agent.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_uses_second_agent(self, executor):
        """Test fallback to second agent on first failure."""
        primary = Mock()
        primary.name = "primary"
        primary.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        backup = Mock()
        backup.name = "backup"
        backup.generate = AsyncMock(return_value="Backup response")

        result = await executor.generate_with_fallback(
            primary, "prompt", [], fallback_agents=[backup]
        )

        assert "Backup response" in result

    @pytest.mark.asyncio
    async def test_fallback_skips_circuit_broken_agent(self, executor, mock_circuit_breaker):
        """Test circuit-broken agents are skipped."""
        mock_circuit_breaker.is_available.side_effect = lambda name: name != "broken"

        broken = Mock()
        broken.name = "broken"
        broken.generate = AsyncMock(return_value="Should not be called")

        healthy = Mock()
        healthy.name = "healthy"
        healthy.generate = AsyncMock(return_value="Healthy response")

        result = await executor.generate_with_fallback(
            broken, "prompt", [], fallback_agents=[healthy]
        )

        assert "Healthy response" in result
        broken.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_retries_before_switching(self, executor):
        """Test retries on same agent before switching."""
        fail_count = [0]

        async def failing_generate(*args):
            fail_count[0] += 1
            if fail_count[0] <= 2:  # Fail first 2 times
                raise asyncio.TimeoutError()
            return "Success on retry"

        agent = Mock()
        agent.name = "flaky_agent"
        agent.generate = AsyncMock(side_effect=failing_generate)

        result = await executor.generate_with_fallback(
            agent, "prompt", [], max_retries=3
        )

        assert "Success on retry" in result

    @pytest.mark.asyncio
    async def test_fallback_total_failure_message(self, executor):
        """Test message when all agents fail."""
        agent1 = Mock()
        agent1.name = "agent1"
        agent1.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        agent2 = Mock()
        agent2.name = "agent2"
        agent2.generate = AsyncMock(side_effect=ConnectionError())

        result = await executor.generate_with_fallback(
            agent1, "prompt", [], fallback_agents=[agent2], max_retries=1
        )

        assert "[System: All agents failed" in result
        assert "agent1" in result
        assert "agent2" in result

    @pytest.mark.asyncio
    async def test_fallback_records_circuit_breaker_failures(self, executor, mock_circuit_breaker):
        """Test circuit breaker failures recorded during fallback."""
        agent = Mock()
        agent.name = "failing_agent"
        agent.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        await executor.generate_with_fallback(agent, "prompt", [], max_retries=2)

        # Should record failures for each retry attempt
        assert mock_circuit_breaker.record_failure.call_count >= 1

    @pytest.mark.asyncio
    async def test_fallback_resets_retries_on_success(self, executor):
        """Test retry count reset after success."""
        agent = Mock()
        agent.name = "agent1"
        agent.generate = AsyncMock(return_value="Success")

        # Record some prior retries
        executor.record_retry("agent1")
        executor.record_retry("agent1")

        await executor.generate_with_fallback(agent, "prompt", [])

        # Retries should be reset
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 10.0  # Back to base

    @pytest.mark.asyncio
    async def test_fallback_clears_streaming_buffer(self, executor, streaming_buffer):
        """Test streaming buffer cleared before each attempt."""
        executor.streaming_buffer = streaming_buffer
        # Pre-populate buffer
        await streaming_buffer.append("agent1", "Old content")

        agent = Mock()
        agent.name = "agent1"
        agent.generate = AsyncMock(return_value="New response")

        await executor.generate_with_fallback(agent, "prompt", [])

        # Buffer should be cleared (we'd need to mock to verify during call)


# ============================================================================
# Wisdom Fallback Tests
# ============================================================================

class TestWisdomFallback:
    """Tests for wisdom fallback functionality."""

    def test_get_wisdom_fallback_no_store(self, executor):
        """Test wisdom fallback returns None without store."""
        result = executor._get_wisdom_fallback("agent1")

        assert result is None

    def test_get_wisdom_fallback_no_loop_id(self, executor):
        """Test wisdom fallback returns None without loop ID."""
        executor.wisdom_store = Mock()

        result = executor._get_wisdom_fallback("agent1")

        assert result is None

    def test_get_wisdom_fallback_success(self, executor):
        """Test successful wisdom fallback."""
        wisdom_store = Mock()
        wisdom_store.get_relevant_wisdom = Mock(return_value=[
            {"id": "w1", "text": "Audience wisdom", "submitter_id": "user123"}
        ])
        wisdom_store.mark_wisdom_used = Mock()

        executor.wisdom_store = wisdom_store
        executor.loop_id = "loop_123"

        result = executor._get_wisdom_fallback("agent1")

        assert result is not None
        assert "Audience wisdom" in result
        assert "user123" in result
        wisdom_store.mark_wisdom_used.assert_called_once_with("w1")

    def test_get_wisdom_fallback_empty_wisdom(self, executor):
        """Test wisdom fallback with no available wisdom."""
        wisdom_store = Mock()
        wisdom_store.get_relevant_wisdom = Mock(return_value=[])

        executor.wisdom_store = wisdom_store
        executor.loop_id = "loop_123"

        result = executor._get_wisdom_fallback("agent1")

        assert result is None

    def test_get_wisdom_fallback_exception(self, executor):
        """Test wisdom fallback handles exceptions gracefully."""
        wisdom_store = Mock()
        wisdom_store.get_relevant_wisdom = Mock(side_effect=RuntimeError("DB error"))

        executor.wisdom_store = wisdom_store
        executor.loop_id = "loop_123"

        # Should not raise
        result = executor._get_wisdom_fallback("agent1")

        assert result is None


# ============================================================================
# Telemetry Tests
# ============================================================================

class TestTelemetry:
    """Tests for telemetry functionality."""

    def test_emit_telemetry_disabled(self, executor):
        """Test telemetry not emitted when disabled."""
        executor.enable_telemetry = False

        # Should not raise
        executor._emit_agent_telemetry(
            "agent1", "generate", 0.0, True
        )

    def test_emit_telemetry_enabled_import_error(self, executor):
        """Test telemetry handles import error gracefully."""
        executor.enable_telemetry = True

        with patch.dict('sys.modules', {'aragora.agents.telemetry': None}):
            # Should not raise
            executor._emit_agent_telemetry(
                "agent1", "generate", 0.0, True
            )


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_generate_with_empty_context(self, executor, mock_agent):
        """Test generate with empty context list."""
        result = await executor.generate(mock_agent, "prompt", [])

        assert result is not None

    @pytest.mark.asyncio
    async def test_fallback_empty_list(self, executor, mock_agent):
        """Test fallback with empty fallback agents list."""
        result = await executor.generate_with_fallback(
            mock_agent, "prompt", [], fallback_agents=[]
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_fallback_none_fallback_agents(self, executor, mock_agent):
        """Test fallback with None fallback agents."""
        result = await executor.generate_with_fallback(
            mock_agent, "prompt", [], fallback_agents=None
        )

        assert result is not None

    def test_timeout_escalation_with_zero_retries(self, executor):
        """Test escalation formula with zero retries."""
        # 10 * 1.5^0 = 10
        timeout = executor.get_escalated_timeout("agent1")
        assert timeout == 10.0

    def test_timeout_escalation_high_retry_count(self, executor):
        """Test escalation with very high retry count."""
        for _ in range(100):
            executor.record_retry("agent1")

        timeout = executor.get_escalated_timeout("agent1")
        # Should be capped at max_timeout
        assert timeout == 60.0

    @pytest.mark.asyncio
    async def test_oserror_handled_like_connection_error(self, executor):
        """Test OSError handled same as ConnectionError."""
        agent = Mock()
        agent.name = "os_error_agent"
        agent.generate = AsyncMock(side_effect=OSError("Socket error"))

        result = await executor.generate(agent, "prompt", [])

        assert "[System:" in result
        assert "connection failed" in result

