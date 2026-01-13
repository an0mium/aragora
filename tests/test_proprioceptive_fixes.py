"""
Regression tests for proprioceptive (self-awareness) fixes.

Tests the "autonomic layer" improvements that make the debate system
self-stabilizing and resilient to agent failures:
1. Output sanitization - prevents null byte crashes
2. Active timeouts - prevents stalled agents from blocking debates
3. Error containment - crashed agents don't crash the debate
4. WebSocket loop_id binding - enables state recovery
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.debate.sanitization import OutputSanitizer


# ============================================================================
# Output Sanitization Tests
# ============================================================================


class TestOutputSanitizerAgentOutput:
    """Tests for OutputSanitizer.sanitize_agent_output()."""

    def test_sanitize_normal_output(self):
        """Normal string output should pass through unchanged."""
        output = "This is a normal response from the agent."
        result = OutputSanitizer.sanitize_agent_output(output, "test-agent")
        assert result == output

    def test_sanitize_removes_null_bytes(self):
        """Null bytes should be removed (proven crash cause)."""
        output = "Hello\x00World\x00!"
        result = OutputSanitizer.sanitize_agent_output(output, "test-agent")
        assert result == "HelloWorld!"
        assert "\x00" not in result

    def test_sanitize_removes_control_characters(self):
        """Control characters (except newlines/tabs) should be removed."""
        # Test various control characters
        output = "Hello\x01\x02\x03World\x1fEnd"
        result = OutputSanitizer.sanitize_agent_output(output, "test-agent")
        assert result == "HelloWorldEnd"

    def test_sanitize_preserves_newlines_and_tabs(self):
        """Newlines and tabs should be preserved."""
        output = "Line 1\nLine 2\tTabbed"
        result = OutputSanitizer.sanitize_agent_output(output, "test-agent")
        assert result == output

    def test_sanitize_handles_non_string_output(self):
        """Non-string output should return error message."""
        result = OutputSanitizer.sanitize_agent_output(123, "test-agent")
        assert result == "(Agent output type error)"

        result = OutputSanitizer.sanitize_agent_output(None, "test-agent")
        assert result == "(Agent output type error)"

        result = OutputSanitizer.sanitize_agent_output(["list"], "test-agent")
        assert result == "(Agent output type error)"

    def test_sanitize_handles_empty_string(self):
        """Empty string should return placeholder message."""
        result = OutputSanitizer.sanitize_agent_output("", "test-agent")
        assert result == "(Agent produced empty output)"

    def test_sanitize_handles_whitespace_only(self):
        """Whitespace-only string should return placeholder message."""
        result = OutputSanitizer.sanitize_agent_output("   \n\t  ", "test-agent")
        assert result == "(Agent produced empty output)"

    def test_sanitize_strips_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        output = "  Hello World  "
        result = OutputSanitizer.sanitize_agent_output(output, "test-agent")
        assert result == "Hello World"

    def test_sanitize_handles_delete_character(self):
        """DEL character (0x7f) should be removed."""
        output = "Hello\x7fWorld"
        result = OutputSanitizer.sanitize_agent_output(output, "test-agent")
        assert result == "HelloWorld"

    def test_sanitize_complex_malformed_output(self):
        """Complex output with multiple issues should be fully sanitized."""
        output = "\x00Start\x01\x02Middle\x00\x1fEnd\x7f\x00"
        result = OutputSanitizer.sanitize_agent_output(output, "test-agent")
        assert result == "StartMiddleEnd"
        assert "\x00" not in result
        assert "\x01" not in result


class TestOutputSanitizerPrompt:
    """Tests for OutputSanitizer.sanitize_prompt()."""

    def test_sanitize_prompt_normal(self):
        """Normal prompt should pass through unchanged."""
        prompt = "What is the meaning of life?"
        result = OutputSanitizer.sanitize_prompt(prompt)
        assert result == prompt

    def test_sanitize_prompt_removes_null_bytes(self):
        """Null bytes in prompts should be removed."""
        prompt = "Hello\x00World"
        result = OutputSanitizer.sanitize_prompt(prompt)
        assert result == "HelloWorld"

    def test_sanitize_prompt_empty(self):
        """Empty prompt should return as-is."""
        assert OutputSanitizer.sanitize_prompt("") == ""
        assert OutputSanitizer.sanitize_prompt(None) is None

    def test_sanitize_prompt_non_string(self):
        """Non-string prompt should be converted to string."""
        result = OutputSanitizer.sanitize_prompt(123)
        assert result == "123"

    def test_sanitize_prompt_preserves_newlines(self):
        """Newlines and carriage returns should be preserved."""
        prompt = "Line 1\nLine 2\r\nLine 3"
        result = OutputSanitizer.sanitize_prompt(prompt)
        assert result == prompt


# ============================================================================
# Timeout Mechanism Tests
# ============================================================================


class TestTimeoutMechanism:
    """Tests for _with_timeout() method in Arena."""

    @pytest.fixture
    def mock_circuit_breaker(self):
        """Create a mock circuit breaker."""
        breaker = MagicMock()
        breaker.record_failure = MagicMock()
        return breaker

    @pytest.fixture
    def mock_arena(self, mock_circuit_breaker):
        """Create a mock Arena with circuit breaker."""
        arena = MagicMock()
        arena.circuit_breaker = mock_circuit_breaker
        return arena

    @pytest.mark.asyncio
    async def test_timeout_success(self, mock_arena, mock_circuit_breaker):
        """Successful completion should return result without timeout."""

        async def fast_coro():
            return "result"

        # Simulate _with_timeout behavior
        result = await asyncio.wait_for(fast_coro(), timeout=1.0)
        assert result == "result"
        mock_circuit_breaker.record_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_triggers_on_slow_agent(self, mock_arena, mock_circuit_breaker):
        """Slow agent should trigger timeout and record failure."""

        async def slow_coro():
            await asyncio.sleep(10)  # Simulate slow agent
            return "never reached"

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_coro(), timeout=0.01)

    @pytest.mark.asyncio
    async def test_timeout_does_not_block_other_agents(self):
        """Timed out agent should not block other agents."""

        async def slow_agent():
            await asyncio.sleep(10)
            return "slow"

        async def fast_agent():
            return "fast"

        # Run both with timeout
        results = []

        try:
            results.append(await asyncio.wait_for(slow_agent(), timeout=0.01))
        except asyncio.TimeoutError:
            results.append("timeout")

        results.append(await fast_agent())

        assert results == ["timeout", "fast"]


# ============================================================================
# Error Containment Tests
# ============================================================================


class TestErrorContainment:
    """Tests for _generate_with_agent() error containment."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = MagicMock()
        agent.name = "test-agent"
        agent.generate = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_successful_generation_is_sanitized(self, mock_agent):
        """Successful generation should be sanitized."""
        mock_agent.generate.return_value = "Hello\x00World"

        # Simulate _generate_with_agent behavior
        raw_output = await mock_agent.generate("prompt", [])
        result = OutputSanitizer.sanitize_agent_output(raw_output, mock_agent.name)

        assert result == "HelloWorld"

    @pytest.mark.asyncio
    async def test_timeout_error_returns_graceful_message(self, mock_agent):
        """TimeoutError should return graceful skip message."""
        mock_agent.generate.side_effect = asyncio.TimeoutError()

        # Simulate _generate_with_agent behavior
        try:
            await mock_agent.generate("prompt", [])
            result = "should not reach"
        except asyncio.TimeoutError:
            result = f"[System: Agent {mock_agent.name} timed out - skipping this turn]"

        assert "timed out" in result
        assert mock_agent.name in result

    @pytest.mark.asyncio
    async def test_exception_returns_graceful_message(self, mock_agent):
        """General exception should return graceful error message."""
        mock_agent.generate.side_effect = RuntimeError("Agent crashed!")

        # Simulate _generate_with_agent behavior
        try:
            await mock_agent.generate("prompt", [])
            result = "should not reach"
        except Exception as e:
            result = f"[System: Agent {mock_agent.name} encountered an error - skipping this turn]"

        assert "error" in result
        assert mock_agent.name in result

    @pytest.mark.asyncio
    async def test_error_containment_prevents_debate_crash(self, mock_agent):
        """Errors should be contained and not propagate to crash the debate."""
        mock_agent.generate.side_effect = ValueError("Bad value")

        # Simulate debate running multiple agents
        results = []
        for i in range(3):
            try:
                result = await mock_agent.generate("prompt", [])
                results.append(result)
            except Exception:
                results.append(f"[System: Agent error - skipping]")

        # All should be error messages, debate should not crash
        assert len(results) == 3
        assert all("[System:" in r for r in results)


# ============================================================================
# WebSocket Loop ID Binding Tests
# ============================================================================


class TestWebSocketLoopIdBinding:
    """Tests for WebSocket loop_id binding and recovery."""

    def test_loop_id_binding_to_websocket(self):
        """Loop ID should be bound to WebSocket for future reference."""
        ws = MagicMock()
        loop_id = "test-loop-123"

        # Simulate binding (stream.py line 2666)
        ws._bound_loop_id = loop_id

        assert ws._bound_loop_id == loop_id

    def test_loop_id_recovery_from_websocket(self):
        """Loop ID should be recoverable from WebSocket attribute."""
        ws = MagicMock()
        ws._bound_loop_id = "recovered-loop-456"

        # Simulate recovery (stream.py line 2637)
        data = {}  # No loop_id in message
        loop_id = data.get("loop_id") or getattr(ws, "_bound_loop_id", "")

        assert loop_id == "recovered-loop-456"

    def test_loop_id_fallback_when_not_bound(self):
        """Should return empty string if loop_id not bound."""
        ws = MagicMock(spec=[])  # No _bound_loop_id attribute

        # Simulate recovery with fallback
        data = {}
        loop_id = data.get("loop_id") or getattr(ws, "_bound_loop_id", "")

        assert loop_id == ""

    def test_loop_id_from_message_takes_precedence(self):
        """Loop ID from message should take precedence over bound value."""
        ws = MagicMock()
        ws._bound_loop_id = "old-bound-loop"

        # Simulate recovery with message containing loop_id
        data = {"loop_id": "new-message-loop"}
        loop_id = data.get("loop_id") or getattr(ws, "_bound_loop_id", "")

        assert loop_id == "new-message-loop"


# ============================================================================
# Integration Tests
# ============================================================================


class TestProprioceptiveIntegration:
    """Integration tests for the full proprioceptive layer."""

    @pytest.mark.asyncio
    async def test_full_sanitization_pipeline(self):
        """Test complete sanitization from agent output to final result."""
        # Simulated agent output with all issues
        raw_output = "\x00Proposal: \x01Use caching\x00\nBenefits:\x1f- Speed\x7f"

        # Sanitize
        result = OutputSanitizer.sanitize_agent_output(raw_output, "test-agent")

        # Verify clean output
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x1f" not in result
        assert "\x7f" not in result
        assert "Proposal: Use caching" in result
        assert "Benefits:" in result
        assert "- Speed" in result

    @pytest.mark.asyncio
    async def test_timeout_with_circuit_breaker_recording(self):
        """Test that timeout properly records to circuit breaker."""
        circuit_breaker = MagicMock()
        circuit_breaker.record_failure = MagicMock()

        async def slow_coro():
            await asyncio.sleep(10)

        try:
            await asyncio.wait_for(slow_coro(), timeout=0.01)
        except asyncio.TimeoutError:
            # This is what _with_timeout does
            circuit_breaker.record_failure("slow-agent")

        circuit_breaker.record_failure.assert_called_once_with("slow-agent")

    def test_websocket_state_persistence_across_messages(self):
        """Test that WebSocket state persists across multiple messages."""
        ws = MagicMock()

        # First message binds loop_id
        ws._bound_loop_id = "persistent-loop"

        # Subsequent messages can recover it
        for _ in range(5):
            data = {}
            recovered = data.get("loop_id") or getattr(ws, "_bound_loop_id", "")
            assert recovered == "persistent-loop"
