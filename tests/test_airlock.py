"""
Tests for the AirlockProxy resilience layer.

Tests cover:
- AirlockMetrics computed properties and serialization
- AirlockConfig default values
- AirlockProxy timeout handling, sanitization, and fallback behavior
- Helper functions wrap_agent() and wrap_agents()
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.airlock import (
    AirlockConfig,
    AirlockMetrics,
    AirlockProxy,
    wrap_agent,
    wrap_agents,
)
from aragora.core import Agent, Critique, Message, Vote


# === Fixtures ===


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent", model: str = "test_model"):
        super().__init__(name, model)
        self.generate_response = "test response"
        self.generate_delay = 0.0
        self.should_raise = False
        self.raise_error = Exception("test error")

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        if self.generate_delay > 0:
            await asyncio.sleep(self.generate_delay)
        if self.should_raise:
            raise self.raise_error
        return self.generate_response

    async def critique(
        self, proposal: str, task: str, context: list[Message] | None = None
    ) -> Critique:
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


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent()


@pytest.fixture
def config():
    """Create a test configuration with short timeouts."""
    return AirlockConfig(
        generate_timeout=1.0,
        critique_timeout=1.0,
        vote_timeout=1.0,
        max_retries=0,
        retry_delay=0.1,
    )


@pytest.fixture
def proxy(mock_agent, config):
    """Create an AirlockProxy with mock agent."""
    return AirlockProxy(mock_agent, config)


# === AirlockMetrics Tests ===


class TestAirlockMetrics:
    """Tests for AirlockMetrics dataclass."""

    def test_default_values(self):
        """Test default initialization values."""
        metrics = AirlockMetrics()
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.timeout_errors == 0
        assert metrics.sanitization_applied == 0
        assert metrics.fallback_responses == 0
        assert metrics.total_latency_ms == 0.0

    def test_success_rate_no_calls(self):
        """Test success rate when no calls made."""
        metrics = AirlockMetrics()
        assert metrics.success_rate == 100.0

    def test_success_rate_all_successful(self):
        """Test success rate when all calls succeed."""
        metrics = AirlockMetrics(total_calls=10, successful_calls=10)
        assert metrics.success_rate == 100.0

    def test_success_rate_partial(self):
        """Test success rate with some failures."""
        metrics = AirlockMetrics(total_calls=10, successful_calls=7)
        assert metrics.success_rate == 70.0

    def test_success_rate_none_successful(self):
        """Test success rate when no calls succeed."""
        metrics = AirlockMetrics(total_calls=10, successful_calls=0)
        assert metrics.success_rate == 0.0

    def test_avg_latency_no_calls(self):
        """Test average latency when no successful calls."""
        metrics = AirlockMetrics()
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_with_calls(self):
        """Test average latency calculation."""
        metrics = AirlockMetrics(successful_calls=5, total_latency_ms=500.0)
        assert metrics.avg_latency_ms == 100.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = AirlockMetrics(
            total_calls=10,
            successful_calls=8,
            timeout_errors=2,
            sanitization_applied=3,
            fallback_responses=1,
            total_latency_ms=800.0,
        )
        result = metrics.to_dict()

        assert result["total_calls"] == 10
        assert result["successful_calls"] == 8
        assert result["timeout_errors"] == 2
        assert result["sanitization_applied"] == 3
        assert result["fallback_responses"] == 1
        assert result["success_rate"] == 80.0
        assert result["avg_latency_ms"] == 100.0

    def test_to_dict_rounds_values(self):
        """Test that to_dict rounds floating point values."""
        metrics = AirlockMetrics(
            total_calls=3,
            successful_calls=1,
            total_latency_ms=333.333,
        )
        result = metrics.to_dict()
        assert result["success_rate"] == 33.33
        assert result["avg_latency_ms"] == 333.33


# === AirlockConfig Tests ===


class TestAirlockConfig:
    """Tests for AirlockConfig dataclass."""

    def test_default_timeouts(self):
        """Test default timeout values."""
        config = AirlockConfig()
        assert config.generate_timeout == 240.0
        assert config.critique_timeout == 180.0
        assert config.vote_timeout == 120.0

    def test_default_retry_settings(self):
        """Test default retry settings."""
        config = AirlockConfig()
        assert config.max_retries == 1
        assert config.retry_delay == 2.0

    def test_default_sanitization_settings(self):
        """Test default sanitization settings."""
        config = AirlockConfig()
        assert config.extract_json is True
        assert config.strip_markdown_fences is True

    def test_default_fallback_settings(self):
        """Test default fallback settings."""
        config = AirlockConfig()
        assert config.fallback_on_timeout is True
        assert config.fallback_on_error is True

    def test_custom_timeouts(self):
        """Test custom timeout values."""
        config = AirlockConfig(
            generate_timeout=60.0,
            critique_timeout=30.0,
            vote_timeout=15.0,
        )
        assert config.generate_timeout == 60.0
        assert config.critique_timeout == 30.0
        assert config.vote_timeout == 15.0


# === AirlockProxy Basic Tests ===


class TestAirlockProxyBasic:
    """Basic tests for AirlockProxy initialization and properties."""

    def test_init_with_default_config(self, mock_agent):
        """Test initialization with default config."""
        proxy = AirlockProxy(mock_agent)
        assert proxy.wrapped_agent is mock_agent
        assert isinstance(proxy._config, AirlockConfig)

    def test_init_with_custom_config(self, mock_agent, config):
        """Test initialization with custom config."""
        proxy = AirlockProxy(mock_agent, config)
        assert proxy._config is config

    def test_metrics_property(self, proxy):
        """Test metrics property returns AirlockMetrics."""
        assert isinstance(proxy.metrics, AirlockMetrics)

    def test_wrapped_agent_property(self, mock_agent, proxy):
        """Test wrapped_agent property returns the original agent."""
        assert proxy.wrapped_agent is mock_agent

    def test_attribute_delegation(self, mock_agent, proxy):
        """Test that attributes are delegated to wrapped agent."""
        assert proxy.name == mock_agent.name
        assert proxy.model == mock_agent.model
        assert proxy.role == mock_agent.role


# === AirlockProxy Generate Tests ===


class TestAirlockProxyGenerate:
    """Tests for AirlockProxy.generate() method."""

    async def test_generate_success(self, proxy):
        """Test successful generation."""
        result = await proxy.generate("test prompt")
        assert result == "test response"
        assert proxy.metrics.total_calls == 1
        assert proxy.metrics.successful_calls == 1

    async def test_generate_timeout_fallback(self, mock_agent, config):
        """Test fallback on timeout."""
        mock_agent.generate_delay = 2.0  # Longer than 1.0 timeout
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("test prompt")
        assert "[Agent test_agent timed out" in result
        assert proxy.metrics.timeout_errors == 1
        assert proxy.metrics.fallback_responses == 1

    async def test_generate_error_fallback(self, mock_agent, config):
        """Test fallback on error."""
        mock_agent.should_raise = True
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("test prompt")
        assert "[Agent test_agent timed out" in result
        assert proxy.metrics.fallback_responses == 1

    async def test_generate_timeout_no_fallback(self, mock_agent):
        """Test that timeout raises when fallback disabled."""
        config = AirlockConfig(
            generate_timeout=0.1,
            fallback_on_timeout=False,
            max_retries=0,
        )
        mock_agent.generate_delay = 1.0
        proxy = AirlockProxy(mock_agent, config)

        with pytest.raises(asyncio.TimeoutError):
            await proxy.generate("test prompt")

    async def test_generate_metrics_tracking(self, proxy):
        """Test that metrics are tracked correctly."""
        await proxy.generate("prompt 1")
        await proxy.generate("prompt 2")
        await proxy.generate("prompt 3")

        assert proxy.metrics.total_calls == 3
        assert proxy.metrics.successful_calls == 3
        assert proxy.metrics.total_latency_ms > 0


# === AirlockProxy Critique Tests ===


class TestAirlockProxyCritique:
    """Tests for AirlockProxy.critique() method."""

    async def test_critique_success(self, proxy):
        """Test successful critique."""
        result = await proxy.critique("proposal", "task")
        assert isinstance(result, Critique)
        assert result.agent == "test_agent"
        assert proxy.metrics.successful_calls == 1

    async def test_critique_timeout_fallback(self, mock_agent, config):
        """Test fallback critique on timeout."""
        mock_agent.generate_delay = 2.0
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.critique("proposal", "task")
        assert isinstance(result, Critique)
        assert result.severity == 0.1  # Fallback severity
        assert "unable to respond in time" in result.issues[0]

    async def test_critique_fallback_converts_dict(self, mock_agent, config):
        """Test that fallback dict is converted to Critique."""
        mock_agent.should_raise = True
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.critique("proposal", "task")
        assert isinstance(result, Critique)
        assert result.agent == "test_agent"


# === AirlockProxy Vote Tests ===


class TestAirlockProxyVote:
    """Tests for AirlockProxy.vote() method."""

    async def test_vote_success(self, mock_agent, config):
        """Test successful vote."""
        # Create a mock vote response
        mock_agent.generate_response = """CHOICE: agent_a
CONFIDENCE: 0.8
CONTINUE: no
REASONING: Best proposal"""
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.vote({"agent_a": "proposal a", "agent_b": "proposal b"}, "task")
        assert isinstance(result, Vote)
        assert proxy.metrics.successful_calls == 1

    async def test_vote_timeout_fallback(self, mock_agent, config):
        """Test fallback vote on timeout."""
        mock_agent.generate_delay = 2.0
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.vote({"agent_a": "proposal a"}, "task")
        assert isinstance(result, Vote)
        assert result.confidence == 0.1  # Fallback confidence
        assert result.continue_debate is False

    async def test_vote_fallback_uses_first_agent(self, mock_agent, config):
        """Test that fallback vote uses first proposal agent."""
        mock_agent.should_raise = True
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.vote({"first_agent": "first", "second_agent": "second"}, "task")
        assert result.choice == "first_agent"


# === AirlockProxy Sanitization Tests ===


class TestAirlockProxySanitization:
    """Tests for AirlockProxy response sanitization."""

    async def test_sanitize_strips_markdown_fences(self, mock_agent, config):
        """Test that markdown code fences are stripped."""
        mock_agent.generate_response = '```json\n{"key": "value"}\n```'
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert "```" not in result
        assert '{"key": "value"}' in result

    async def test_sanitize_extracts_json_from_text(self, mock_agent, config):
        """Test JSON extraction from surrounding text."""
        mock_agent.generate_response = 'Here is the JSON: {"key": "value"} and more text'
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert result == '{"key": "value"}'
        assert proxy.metrics.sanitization_applied == 1

    async def test_sanitize_removes_control_chars(self, mock_agent, config):
        """Test removal of control characters."""
        mock_agent.generate_response = "test\x00response\x1fhere"
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "testresponsehere" in result

    async def test_sanitize_preserves_newlines(self, mock_agent, config):
        """Test that newlines are preserved."""
        mock_agent.generate_response = "line1\nline2\ttab"
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert "\n" in result
        assert "\t" in result

    async def test_sanitize_disabled_extract_json(self, mock_agent):
        """Test with JSON extraction disabled."""
        config = AirlockConfig(extract_json=False)
        mock_agent.generate_response = 'prefix {"key": "value"} suffix'
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert "prefix" in result
        assert "suffix" in result

    async def test_sanitize_disabled_strip_fences(self, mock_agent):
        """Test with markdown fence stripping disabled."""
        config = AirlockConfig(strip_markdown_fences=False, extract_json=False)
        mock_agent.generate_response = '```json\n{"key": "value"}\n```'
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert "```json" in result

    async def test_sanitize_handles_json_array(self, mock_agent, config):
        """Test JSON array extraction."""
        mock_agent.generate_response = "List: [1, 2, 3] end"
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert result == "[1, 2, 3]"

    async def test_sanitize_invalid_json_preserved(self, mock_agent, config):
        """Test that invalid JSON doesn't break sanitization."""
        mock_agent.generate_response = "This has {broken: json} here"
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert "broken" in result

    async def test_sanitize_empty_response(self, mock_agent, config):
        """Test handling of empty response."""
        mock_agent.generate_response = ""
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert result == ""


# === AirlockProxy Retry Tests ===


class TestAirlockProxyRetry:
    """Tests for AirlockProxy retry behavior."""

    async def test_retry_on_timeout(self, mock_agent):
        """Test that retries happen on timeout."""
        config = AirlockConfig(
            generate_timeout=0.1,
            max_retries=2,
            retry_delay=0.01,
        )
        mock_agent.generate_delay = 0.5  # Always timeout

        proxy = AirlockProxy(mock_agent, config)
        result = await proxy.generate("prompt")

        # Should have tried 3 times (initial + 2 retries)
        assert proxy.metrics.timeout_errors == 3
        assert proxy.metrics.fallback_responses == 1

    async def test_retry_succeeds_second_attempt(self, mock_agent):
        """Test successful retry after initial failure."""
        config = AirlockConfig(
            generate_timeout=1.0,
            max_retries=1,
            retry_delay=0.01,
        )
        attempt_count = 0

        original_generate = mock_agent.generate

        async def flaky_generate(prompt, context=None):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise Exception("First attempt fails")
            return await original_generate(prompt, context)

        mock_agent.generate = flaky_generate
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert result == "test response"
        assert attempt_count == 2


# === Helper Function Tests ===


class TestHelperFunctions:
    """Tests for wrap_agent() and wrap_agents() helper functions."""

    def test_wrap_agent_basic(self, mock_agent):
        """Test basic agent wrapping."""
        proxy = wrap_agent(mock_agent)
        assert isinstance(proxy, AirlockProxy)
        assert proxy.wrapped_agent is mock_agent

    def test_wrap_agent_with_config(self, mock_agent, config):
        """Test agent wrapping with custom config."""
        proxy = wrap_agent(mock_agent, config)
        assert proxy._config is config

    def test_wrap_agents_empty_list(self):
        """Test wrapping empty list of agents."""
        result = wrap_agents([])
        assert result == []

    def test_wrap_agents_multiple(self):
        """Test wrapping multiple agents."""
        agents = [MockAgent(f"agent_{i}") for i in range(3)]
        proxies = wrap_agents(agents)

        assert len(proxies) == 3
        for i, proxy in enumerate(proxies):
            assert isinstance(proxy, AirlockProxy)
            assert proxy.name == f"agent_{i}"

    def test_wrap_agents_with_config(self, config):
        """Test wrapping multiple agents with shared config."""
        agents = [MockAgent(f"agent_{i}") for i in range(2)]
        proxies = wrap_agents(agents, config)

        for proxy in proxies:
            assert proxy._config is config


# === Edge Case Tests ===


class TestEdgeCases:
    """Edge case tests for AirlockProxy."""

    async def test_concurrent_calls(self, mock_agent, config):
        """Test concurrent generate calls."""
        proxy = AirlockProxy(mock_agent, config)

        results = await asyncio.gather(
            proxy.generate("prompt 1"),
            proxy.generate("prompt 2"),
            proxy.generate("prompt 3"),
        )

        assert len(results) == 3
        assert proxy.metrics.total_calls == 3
        assert proxy.metrics.successful_calls == 3

    async def test_nested_json_extraction(self, mock_agent, config):
        """Test extraction of nested JSON."""
        mock_agent.generate_response = 'Response: {"outer": {"inner": "value"}}'
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.generate("prompt")
        assert '{"outer": {"inner": "value"}}' == result

    async def test_context_passed_through(self, mock_agent, config):
        """Test that context is passed to underlying agent."""
        received_context = None

        async def capture_generate(prompt, context=None):
            nonlocal received_context
            received_context = context
            return "response"

        mock_agent.generate = capture_generate
        proxy = AirlockProxy(mock_agent, config)

        messages = [Message(role="user", agent="test", content="hello")]
        await proxy.generate("prompt", context=messages)

        assert received_context == messages

    async def test_vote_empty_proposals(self, mock_agent, config):
        """Test vote with empty proposals dict."""
        mock_agent.should_raise = True
        proxy = AirlockProxy(mock_agent, config)

        result = await proxy.vote({}, "task")
        assert result.choice == "unknown"
