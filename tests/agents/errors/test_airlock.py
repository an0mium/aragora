"""
Tests for Airlock resilience layer.

Tests cover:
- AirlockMetrics dataclass
- AirlockConfig dataclass
- AirlockProxy wrapper
- Timeout handling
- Response sanitization
- Fallback responses
- Metrics collection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.airlock import (
    AirlockConfig,
    AirlockMetrics,
    AirlockProxy,
    wrap_agent,
    wrap_agents,
)


# ============================================================================
# Mock Agent Fixture
# ============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "mock-agent"
    model: str = "mock-model"

    async def generate(self, prompt: str, context: Optional[list] = None) -> str:
        """Generate a response."""
        return f"Response to: {prompt[:50]}"

    async def critique(self, proposal: str, task: str, context: Optional[list] = None) -> dict:
        """Critique a proposal."""
        return {
            "agent": self.name,
            "target_agent": "other",
            "target_content": proposal[:100],
            "issues": [],
            "suggestions": [],
            "severity": 0.5,
            "reasoning": "Mock critique",
        }

    async def vote(self, proposals: dict[str, str], task: str) -> dict:
        """Vote on proposals."""
        first_agent = next(iter(proposals.keys()), "unknown")
        return {
            "agent": self.name,
            "choice": first_agent,
            "reasoning": "Mock vote",
            "confidence": 0.8,
            "continue_debate": False,
        }


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    return MockAgent()


@pytest.fixture
def slow_agent():
    """Create a mock agent that takes a long time to respond."""
    agent = MockAgent(name="slow-agent")

    async def slow_generate(prompt: str, context: Optional[list] = None) -> str:
        await asyncio.sleep(10)  # Longer than default timeout in tests
        return "Slow response"

    agent.generate = slow_generate
    return agent


@pytest.fixture
def failing_agent():
    """Create a mock agent that raises errors."""
    agent = MockAgent(name="failing-agent")

    async def failing_generate(prompt: str, context: Optional[list] = None) -> str:
        raise ConnectionError("Connection refused")

    agent.generate = failing_generate
    return agent


# ============================================================================
# AirlockMetrics Tests
# ============================================================================


class TestAirlockMetrics:
    """Tests for AirlockMetrics dataclass."""

    def test_default_metrics(self):
        """Test default metric values."""
        metrics = AirlockMetrics()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.timeout_errors == 0
        assert metrics.sanitization_applied == 0
        assert metrics.fallback_responses == 0
        assert metrics.total_latency_ms == 0.0

    def test_success_rate_empty(self):
        """Test success rate with no calls."""
        metrics = AirlockMetrics()
        assert metrics.success_rate == 100.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = AirlockMetrics(total_calls=10, successful_calls=8)
        assert metrics.success_rate == 80.0

    def test_success_rate_all_successful(self):
        """Test success rate with all successful."""
        metrics = AirlockMetrics(total_calls=5, successful_calls=5)
        assert metrics.success_rate == 100.0

    def test_success_rate_all_failed(self):
        """Test success rate with all failed."""
        metrics = AirlockMetrics(total_calls=5, successful_calls=0)
        assert metrics.success_rate == 0.0

    def test_avg_latency_empty(self):
        """Test average latency with no calls."""
        metrics = AirlockMetrics()
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = AirlockMetrics(successful_calls=4, total_latency_ms=100.0)
        assert metrics.avg_latency_ms == 25.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = AirlockMetrics(
            total_calls=10,
            successful_calls=8,
            timeout_errors=1,
            sanitization_applied=3,
            fallback_responses=1,
            total_latency_ms=500.0,
        )

        result = metrics.to_dict()

        assert result["total_calls"] == 10
        assert result["successful_calls"] == 8
        assert result["timeout_errors"] == 1
        assert result["sanitization_applied"] == 3
        assert result["fallback_responses"] == 1
        assert result["success_rate"] == 80.0
        assert result["avg_latency_ms"] == 62.5


# ============================================================================
# AirlockConfig Tests
# ============================================================================


class TestAirlockConfig:
    """Tests for AirlockConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AirlockConfig()

        assert config.generate_timeout == 240.0
        assert config.critique_timeout == 180.0
        assert config.vote_timeout == 120.0
        assert config.max_retries == 1
        assert config.retry_delay == 2.0
        assert config.extract_json is True
        assert config.strip_markdown_fences is True
        assert config.fallback_on_timeout is True
        assert config.fallback_on_error is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AirlockConfig(
            generate_timeout=60.0,
            max_retries=3,
            fallback_on_timeout=False,
        )

        assert config.generate_timeout == 60.0
        assert config.max_retries == 3
        assert config.fallback_on_timeout is False


# ============================================================================
# AirlockProxy Basic Tests
# ============================================================================


class TestAirlockProxyBasic:
    """Tests for AirlockProxy basic functionality."""

    def test_proxy_creation(self, mock_agent):
        """Test creating an airlock proxy."""
        proxy = AirlockProxy(mock_agent)

        assert proxy.wrapped_agent is mock_agent
        assert proxy.metrics.total_calls == 0

    def test_proxy_with_custom_config(self, mock_agent):
        """Test creating proxy with custom config."""
        config = AirlockConfig(generate_timeout=30.0)
        proxy = AirlockProxy(mock_agent, config)

        assert proxy._config.generate_timeout == 30.0

    def test_proxy_delegates_attributes(self, mock_agent):
        """Test proxy delegates attribute access to wrapped agent."""
        proxy = AirlockProxy(mock_agent)

        assert proxy.name == "mock-agent"
        assert proxy.model == "mock-model"

    def test_metrics_property(self, mock_agent):
        """Test metrics property returns metrics object."""
        proxy = AirlockProxy(mock_agent)
        metrics = proxy.metrics

        assert isinstance(metrics, AirlockMetrics)


# ============================================================================
# AirlockProxy Generate Tests
# ============================================================================


class TestAirlockProxyGenerate:
    """Tests for AirlockProxy.generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_agent):
        """Test successful generate call."""
        proxy = AirlockProxy(mock_agent)

        result = await proxy.generate("Hello, world!")

        assert "Response to: Hello, world!" in result
        assert proxy.metrics.total_calls == 1
        assert proxy.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_generate_with_context(self, mock_agent):
        """Test generate with context."""
        proxy = AirlockProxy(mock_agent)

        result = await proxy.generate("Hello", context=[])

        assert "Response to: Hello" in result

    @pytest.mark.asyncio
    async def test_generate_timeout_fallback(self, slow_agent):
        """Test generate falls back on timeout."""
        config = AirlockConfig(generate_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(slow_agent, config)

        result = await proxy.generate("Test prompt")

        assert "timed out" in result.lower()
        assert proxy.metrics.timeout_errors >= 1
        assert proxy.metrics.fallback_responses == 1

    @pytest.mark.asyncio
    async def test_generate_error_fallback(self, failing_agent):
        """Test generate falls back on error."""
        config = AirlockConfig(max_retries=0)
        proxy = AirlockProxy(failing_agent, config)

        result = await proxy.generate("Test prompt")

        assert "timed out" in result.lower() or "unable to generate" in result.lower()
        assert proxy.metrics.fallback_responses >= 1

    @pytest.mark.asyncio
    async def test_generate_tracks_latency(self, mock_agent):
        """Test generate tracks latency metrics."""
        proxy = AirlockProxy(mock_agent)

        await proxy.generate("Hello")

        assert proxy.metrics.total_latency_ms > 0


# ============================================================================
# AirlockProxy Critique Tests
# ============================================================================


class TestAirlockProxyCritique:
    """Tests for AirlockProxy.critique method."""

    @pytest.mark.asyncio
    async def test_critique_success(self, mock_agent):
        """Test successful critique call."""
        proxy = AirlockProxy(mock_agent)

        result = await proxy.critique(
            proposal="Test proposal", task="Test task", target_agent="test-target"
        )

        # Result should be a Critique object (imported dynamically)
        assert hasattr(result, "agent")
        assert proxy.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_critique_timeout_fallback(self):
        """Test critique falls back on timeout."""
        agent = MockAgent()

        async def slow_critique(proposal: str, task: str, context=None) -> dict:
            await asyncio.sleep(10)
            return {}

        agent.critique = slow_critique

        config = AirlockConfig(critique_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.critique(
            proposal="Test proposal", task="Test task", target_agent="test-target"
        )

        assert "timeout" in result.reasoning.lower() or "fallback" in result.reasoning.lower()
        assert proxy.metrics.fallback_responses >= 1


# ============================================================================
# AirlockProxy Vote Tests
# ============================================================================


class TestAirlockProxyVote:
    """Tests for AirlockProxy.vote method."""

    @pytest.mark.asyncio
    async def test_vote_success(self, mock_agent):
        """Test successful vote call."""
        proxy = AirlockProxy(mock_agent)
        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}

        result = await proxy.vote(proposals, "Test task")

        assert hasattr(result, "choice")
        assert proxy.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_vote_timeout_fallback(self):
        """Test vote falls back on timeout."""
        agent = MockAgent()

        async def slow_vote(proposals: dict, task: str) -> dict:
            await asyncio.sleep(10)
            return {}

        agent.vote = slow_vote

        config = AirlockConfig(vote_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(agent, config)

        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        result = await proxy.vote(proposals, "Test task")

        assert result.choice == "agent1"  # Falls back to first proposal
        assert proxy.metrics.fallback_responses >= 1


# ============================================================================
# AirlockProxy Retry Tests
# ============================================================================


class TestAirlockProxyRetry:
    """Tests for AirlockProxy retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test retry on connection error."""
        agent = MockAgent()
        call_count = 0

        async def flaky_generate(prompt: str, context=None) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Temporary failure")
            return "Success on retry"

        agent.generate = flaky_generate

        config = AirlockConfig(max_retries=1, retry_delay=0.01)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test")

        assert result == "Success on retry"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_value_error(self):
        """Test no retry on validation errors."""
        agent = MockAgent()
        call_count = 0

        async def validation_error(prompt: str, context=None) -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        agent.generate = validation_error

        config = AirlockConfig(max_retries=3, fallback_on_error=True)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test")

        # Should not retry ValueError (non-retryable)
        assert call_count == 1
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test retry exhaustion falls back."""
        agent = MockAgent()
        call_count = 0

        async def always_fail(prompt: str, context=None) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        agent.generate = always_fail

        config = AirlockConfig(max_retries=2, retry_delay=0.01, fallback_on_error=True)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test")

        assert call_count == 3  # Initial + 2 retries
        assert proxy.metrics.fallback_responses == 1


# ============================================================================
# AirlockProxy Sanitization Tests
# ============================================================================


class TestAirlockProxySanitization:
    """Tests for AirlockProxy response sanitization."""

    @pytest.mark.asyncio
    async def test_strip_markdown_fences(self):
        """Test stripping markdown code fences."""
        agent = MockAgent()

        async def markdown_response(prompt: str, context=None) -> str:
            return '```json\n{"key": "value"}\n```'

        agent.generate = markdown_response

        config = AirlockConfig(strip_markdown_fences=True)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test")

        assert "```" not in result
        assert '{"key": "value"}' in result
        assert proxy.metrics.sanitization_applied == 1

    @pytest.mark.asyncio
    async def test_extract_json_from_text(self):
        """Test extracting JSON from surrounding text."""
        agent = MockAgent()

        async def json_in_text(prompt: str, context=None) -> str:
            return 'Here is the data: {"name": "test", "value": 42} Hope that helps!'

        agent.generate = json_in_text

        config = AirlockConfig(extract_json=True)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test")

        assert result == '{"name": "test", "value": 42}'
        assert proxy.metrics.sanitization_applied == 1

    @pytest.mark.asyncio
    async def test_extract_json_array(self):
        """Test extracting JSON array from text."""
        agent = MockAgent()

        async def array_in_text(prompt: str, context=None) -> str:
            return "The list is: [1, 2, 3] as requested."

        agent.generate = array_in_text

        config = AirlockConfig(extract_json=True)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test")

        assert result == "[1, 2, 3]"

    @pytest.mark.asyncio
    async def test_remove_control_characters(self):
        """Test removing control characters."""
        agent = MockAgent()

        async def control_chars(prompt: str, context=None) -> str:
            return "Hello\x00World\x08Test"

        agent.generate = control_chars

        proxy = AirlockProxy(agent)

        result = await proxy.generate("Test")

        assert "\x00" not in result
        assert "\x08" not in result
        assert "HelloWorldTest" in result

    @pytest.mark.asyncio
    async def test_no_sanitization_when_disabled(self):
        """Test no sanitization when disabled."""
        agent = MockAgent()

        async def markdown_response(prompt: str, context=None) -> str:
            return '```json\n{"key": "value"}\n```'

        agent.generate = markdown_response

        config = AirlockConfig(strip_markdown_fences=False, extract_json=False)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test")

        # Control chars still stripped, but fences remain
        assert "```json" in result
        # Fences might be partially stripped due to the logic
        assert proxy.metrics.sanitization_applied == 0


# ============================================================================
# wrap_agent and wrap_agents Tests
# ============================================================================


class TestWrapFunctions:
    """Tests for wrap_agent and wrap_agents functions."""

    def test_wrap_agent(self, mock_agent):
        """Test wrap_agent function."""
        proxy = wrap_agent(mock_agent)

        assert isinstance(proxy, AirlockProxy)
        assert proxy.wrapped_agent is mock_agent

    def test_wrap_agent_with_config(self, mock_agent):
        """Test wrap_agent with custom config."""
        config = AirlockConfig(generate_timeout=60.0)
        proxy = wrap_agent(mock_agent, config)

        assert proxy._config.generate_timeout == 60.0

    def test_wrap_agents(self, mock_agent):
        """Test wrap_agents function."""
        agents = [mock_agent, MockAgent(name="agent2")]
        proxies = wrap_agents(agents)

        assert len(proxies) == 2
        assert all(isinstance(p, AirlockProxy) for p in proxies)
        assert proxies[0].name == "mock-agent"
        assert proxies[1].name == "agent2"

    def test_wrap_agents_with_config(self, mock_agent):
        """Test wrap_agents with shared config."""
        config = AirlockConfig(max_retries=5)
        agents = [mock_agent, MockAgent(name="agent2")]
        proxies = wrap_agents(agents, config)

        assert all(p._config.max_retries == 5 for p in proxies)


# ============================================================================
# AirlockProxy Fallback Generation Tests
# ============================================================================


class TestAirlockProxyFallbacks:
    """Tests for AirlockProxy fallback response generation."""

    @pytest.mark.asyncio
    async def test_generate_fallback_contains_agent_name(self, slow_agent):
        """Test generate fallback mentions agent name."""
        config = AirlockConfig(generate_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(slow_agent, config)

        result = await proxy.generate("Test prompt")

        assert "slow-agent" in result

    @pytest.mark.asyncio
    async def test_critique_fallback_has_required_fields(self):
        """Test critique fallback has all required fields."""
        agent = MockAgent()

        async def slow_critique(proposal: str, task: str, context=None) -> dict:
            await asyncio.sleep(10)
            return {}

        agent.critique = slow_critique

        config = AirlockConfig(critique_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.critique(
            proposal="Test proposal", task="Test task", target_agent="target"
        )

        assert hasattr(result, "agent")
        assert hasattr(result, "issues")
        assert hasattr(result, "severity")
        assert hasattr(result, "reasoning")

    @pytest.mark.asyncio
    async def test_vote_fallback_picks_first_proposal(self):
        """Test vote fallback picks first proposal."""
        agent = MockAgent()

        async def slow_vote(proposals: dict, task: str) -> dict:
            await asyncio.sleep(10)
            return {}

        agent.vote = slow_vote

        config = AirlockConfig(vote_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(agent, config)

        # Use ordered dict-like behavior
        proposals = {"first_agent": "First proposal", "second_agent": "Second proposal"}
        result = await proxy.vote(proposals, "Test task")

        assert result.choice == "first_agent"
        assert result.confidence == 0.1  # Low confidence fallback


# ============================================================================
# AirlockProxy Error Handling Edge Cases
# ============================================================================


class TestAirlockProxyEdgeCases:
    """Tests for AirlockProxy edge cases."""

    @pytest.mark.asyncio
    async def test_empty_response_sanitization(self, mock_agent):
        """Test sanitization handles empty response."""

        async def empty_response(prompt: str, context=None) -> str:
            return ""

        mock_agent.generate = empty_response
        proxy = AirlockProxy(mock_agent)

        result = await proxy.generate("Test")

        assert result == ""

    @pytest.mark.asyncio
    async def test_non_string_response_not_sanitized(self, mock_agent):
        """Test non-string responses are not sanitized."""
        # For critique, the result is a dict, not sanitized as string
        proxy = AirlockProxy(mock_agent)

        result = await proxy.critique("Test proposal", "Test task")

        # Result should be a Critique object
        assert hasattr(result, "agent")

    @pytest.mark.asyncio
    async def test_fallback_disabled_raises(self):
        """Test that errors are raised when fallback is disabled."""
        agent = MockAgent()

        async def always_fail(prompt: str, context=None) -> str:
            raise RuntimeError("Expected failure")

        agent.generate = always_fail

        config = AirlockConfig(max_retries=0, fallback_on_error=False)
        proxy = AirlockProxy(agent, config)

        with pytest.raises(RuntimeError, match="Expected failure"):
            await proxy.generate("Test")

    @pytest.mark.asyncio
    async def test_timeout_raises_when_fallback_disabled(self, slow_agent):
        """Test timeout raises when fallback disabled."""
        config = AirlockConfig(generate_timeout=0.1, max_retries=0, fallback_on_timeout=False)
        proxy = AirlockProxy(slow_agent, config)

        with pytest.raises(asyncio.TimeoutError):
            await proxy.generate("Test")
