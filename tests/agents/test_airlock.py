"""
Tests for the Airlock Resilience Layer.

Tests cover:
- AirlockMetrics dataclass
- AirlockConfig configuration
- AirlockProxy wrapper class
- Timeout handling
- Response sanitization
- Fallback responses
- Metrics collection
- Malformed output handling
- wrap_agent() and wrap_agents() functions
- Error handling and retries
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_can_import_module(self):
        """Module can be imported."""
        from aragora.agents import airlock

        assert airlock is not None

    def test_airlock_metrics_in_all(self):
        """AirlockMetrics is exported in __all__."""
        from aragora.agents.airlock import __all__

        assert "AirlockMetrics" in __all__

    def test_airlock_config_in_all(self):
        """AirlockConfig is exported in __all__."""
        from aragora.agents.airlock import __all__

        assert "AirlockConfig" in __all__

    def test_airlock_proxy_in_all(self):
        """AirlockProxy is exported in __all__."""
        from aragora.agents.airlock import __all__

        assert "AirlockProxy" in __all__

    def test_wrap_agent_in_all(self):
        """wrap_agent is exported in __all__."""
        from aragora.agents.airlock import __all__

        assert "wrap_agent" in __all__

    def test_wrap_agents_in_all(self):
        """wrap_agents is exported in __all__."""
        from aragora.agents.airlock import __all__

        assert "wrap_agents" in __all__


# =============================================================================
# AirlockMetrics Tests
# =============================================================================


class TestAirlockMetricsInit:
    """Test AirlockMetrics initialization."""

    def test_default_values(self):
        """AirlockMetrics initializes with correct defaults."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics()

        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.timeout_errors == 0
        assert metrics.sanitization_applied == 0
        assert metrics.fallback_responses == 0
        assert metrics.total_latency_ms == 0.0

    def test_custom_values(self):
        """AirlockMetrics can be initialized with custom values."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics(
            total_calls=10,
            successful_calls=8,
            timeout_errors=2,
            sanitization_applied=3,
            fallback_responses=1,
            total_latency_ms=500.0,
        )

        assert metrics.total_calls == 10
        assert metrics.successful_calls == 8
        assert metrics.timeout_errors == 2
        assert metrics.sanitization_applied == 3
        assert metrics.fallback_responses == 1
        assert metrics.total_latency_ms == 500.0


class TestAirlockMetricsSuccessRate:
    """Test AirlockMetrics success_rate property."""

    def test_success_rate_zero_calls(self):
        """Success rate is 100% with zero calls."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics()

        assert metrics.success_rate == 100.0

    def test_success_rate_all_successful(self):
        """Success rate is 100% when all calls succeed."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics(total_calls=10, successful_calls=10)

        assert metrics.success_rate == 100.0

    def test_success_rate_partial(self):
        """Success rate is calculated correctly for partial success."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics(total_calls=10, successful_calls=7)

        assert metrics.success_rate == 70.0

    def test_success_rate_all_failed(self):
        """Success rate is 0% when all calls fail."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics(total_calls=10, successful_calls=0)

        assert metrics.success_rate == 0.0


class TestAirlockMetricsAvgLatency:
    """Test AirlockMetrics avg_latency_ms property."""

    def test_avg_latency_zero_calls(self):
        """Average latency is 0 with zero successful calls."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics()

        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_calculated(self):
        """Average latency is calculated correctly."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics(successful_calls=5, total_latency_ms=1000.0)

        assert metrics.avg_latency_ms == 200.0


class TestAirlockMetricsToDict:
    """Test AirlockMetrics to_dict method."""

    def test_to_dict_contains_all_fields(self):
        """to_dict returns all expected fields."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics(
            total_calls=10,
            successful_calls=8,
            timeout_errors=2,
            sanitization_applied=3,
            fallback_responses=1,
            total_latency_ms=800.0,
        )

        result = metrics.to_dict()

        assert "total_calls" in result
        assert "successful_calls" in result
        assert "timeout_errors" in result
        assert "sanitization_applied" in result
        assert "fallback_responses" in result
        assert "success_rate" in result
        assert "avg_latency_ms" in result

    def test_to_dict_values_correct(self):
        """to_dict returns correct values."""
        from aragora.agents.airlock import AirlockMetrics

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
        assert result["success_rate"] == 80.0
        assert result["avg_latency_ms"] == 100.0

    def test_to_dict_rounds_floats(self):
        """to_dict rounds float values to 2 decimal places."""
        from aragora.agents.airlock import AirlockMetrics

        metrics = AirlockMetrics(
            total_calls=3,
            successful_calls=1,
            total_latency_ms=100.12345,
        )

        result = metrics.to_dict()

        assert result["success_rate"] == round(1 / 3 * 100, 2)
        assert result["avg_latency_ms"] == round(100.12345, 2)


# =============================================================================
# AirlockConfig Tests
# =============================================================================


class TestAirlockConfigDefaults:
    """Test AirlockConfig default values."""

    def test_default_generate_timeout(self):
        """Default generate timeout is 240 seconds."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.generate_timeout == 240.0

    def test_default_critique_timeout(self):
        """Default critique timeout is 180 seconds."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.critique_timeout == 180.0

    def test_default_vote_timeout(self):
        """Default vote timeout is 120 seconds."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.vote_timeout == 120.0

    def test_default_max_retries(self):
        """Default max retries is 1."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.max_retries == 1

    def test_default_retry_delay(self):
        """Default retry delay is 2.0 seconds."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.retry_delay == 2.0

    def test_default_extract_json(self):
        """Default extract_json is True."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.extract_json is True

    def test_default_strip_markdown_fences(self):
        """Default strip_markdown_fences is True."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.strip_markdown_fences is True

    def test_default_fallback_on_timeout(self):
        """Default fallback_on_timeout is True."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.fallback_on_timeout is True

    def test_default_fallback_on_error(self):
        """Default fallback_on_error is True."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig()

        assert config.fallback_on_error is True


class TestAirlockConfigCustom:
    """Test AirlockConfig with custom values."""

    def test_custom_timeouts(self):
        """Custom timeouts can be set."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig(
            generate_timeout=60.0,
            critique_timeout=45.0,
            vote_timeout=30.0,
        )

        assert config.generate_timeout == 60.0
        assert config.critique_timeout == 45.0
        assert config.vote_timeout == 30.0

    def test_custom_retries(self):
        """Custom retry settings can be set."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig(
            max_retries=5,
            retry_delay=5.0,
        )

        assert config.max_retries == 5
        assert config.retry_delay == 5.0

    def test_disable_sanitization(self):
        """Sanitization can be disabled."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig(
            extract_json=False,
            strip_markdown_fences=False,
        )

        assert config.extract_json is False
        assert config.strip_markdown_fences is False

    def test_disable_fallback(self):
        """Fallback can be disabled."""
        from aragora.agents.airlock import AirlockConfig

        config = AirlockConfig(
            fallback_on_timeout=False,
            fallback_on_error=False,
        )

        assert config.fallback_on_timeout is False
        assert config.fallback_on_error is False


# =============================================================================
# AirlockProxy Tests
# =============================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent", model: str = "test-model"):
        self.name = name
        self.model = model
        self.role = "proposer"

    async def generate(self, prompt: str, context: list | None = None) -> str:
        return f"Response to: {prompt}"

    async def critique(self, proposal: str, task: str, context: list | None = None) -> Any:
        from aragora.core_types import Critique

        return Critique(
            agent=self.name,
            target_agent="target",
            target_content=proposal[:100],
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=5.0,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Any:
        from aragora.core_types import Vote

        first_agent = next(iter(proposals.keys()))
        return Vote(
            agent=self.name,
            choice=first_agent,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=True,
        )


class TestAirlockProxyInit:
    """Test AirlockProxy initialization."""

    def test_init_with_agent(self):
        """AirlockProxy initializes with agent."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        assert proxy._agent is agent
        assert proxy._config is not None
        assert proxy._metrics is not None

    def test_init_with_custom_config(self):
        """AirlockProxy initializes with custom config."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()
        config = AirlockConfig(generate_timeout=30.0)
        proxy = AirlockProxy(agent, config)

        assert proxy._config.generate_timeout == 30.0

    def test_init_uses_default_config(self):
        """AirlockProxy uses default config when none provided."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        # Should have default values
        assert proxy._config.generate_timeout == AirlockConfig().generate_timeout


class TestAirlockProxyAttributeAccess:
    """Test AirlockProxy attribute delegation."""

    def test_delegates_name(self):
        """AirlockProxy delegates name attribute."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent(name="my_agent")
        proxy = AirlockProxy(agent)

        assert proxy.name == "my_agent"

    def test_delegates_model(self):
        """AirlockProxy delegates model attribute."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent(model="gpt-4")
        proxy = AirlockProxy(agent)

        assert proxy.model == "gpt-4"

    def test_delegates_role(self):
        """AirlockProxy delegates role attribute."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.role = "critic"
        proxy = AirlockProxy(agent)

        assert proxy.role == "critic"

    def test_wrapped_agent_property(self):
        """AirlockProxy exposes wrapped_agent property."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        assert proxy.wrapped_agent is agent

    def test_metrics_property(self):
        """AirlockProxy exposes metrics property."""
        from aragora.agents.airlock import AirlockMetrics, AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        assert isinstance(proxy.metrics, AirlockMetrics)


class TestAirlockProxyGenerate:
    """Test AirlockProxy generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """AirlockProxy.generate returns response on success."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        result = await proxy.generate("Test prompt")

        assert "Response to: Test prompt" in result

    @pytest.mark.asyncio
    async def test_generate_updates_metrics(self):
        """AirlockProxy.generate updates metrics on success."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        await proxy.generate("Test prompt")

        assert proxy.metrics.total_calls == 1
        assert proxy.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """AirlockProxy.generate works with context."""
        from aragora.agents.airlock import AirlockProxy
        from aragora.core_types import Message

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="Response with context")
        proxy = AirlockProxy(agent)

        context = [Message(agent="other", content="Prior message", round=1, role="proposer")]
        result = await proxy.generate("Test prompt", context)

        assert result == "Response with context"
        agent.generate.assert_called_once_with("Test prompt", context)


class TestAirlockProxyCritique:
    """Test AirlockProxy critique method."""

    @pytest.mark.asyncio
    async def test_critique_success(self):
        """AirlockProxy.critique returns Critique on success."""
        from aragora.agents.airlock import AirlockProxy
        from aragora.core_types import Critique

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        result = await proxy.critique("Test proposal", "Test task")

        assert isinstance(result, Critique)
        assert result.agent == "test_agent"

    @pytest.mark.asyncio
    async def test_critique_updates_metrics(self):
        """AirlockProxy.critique updates metrics."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        await proxy.critique("Test proposal", "Test task")

        assert proxy.metrics.total_calls == 1
        assert proxy.metrics.successful_calls == 1


class TestAirlockProxyVote:
    """Test AirlockProxy vote method."""

    @pytest.mark.asyncio
    async def test_vote_success(self):
        """AirlockProxy.vote returns Vote on success."""
        from aragora.agents.airlock import AirlockProxy
        from aragora.core_types import Vote

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        result = await proxy.vote(proposals, "Test task")

        assert isinstance(result, Vote)
        assert result.agent == "test_agent"

    @pytest.mark.asyncio
    async def test_vote_updates_metrics(self):
        """AirlockProxy.vote updates metrics."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        proposals = {"agent1": "Proposal 1"}
        await proxy.vote(proposals, "Test task")

        assert proxy.metrics.total_calls == 1
        assert proxy.metrics.successful_calls == 1


# =============================================================================
# Timeout Handling Tests
# =============================================================================


class TestTimeoutHandling:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback(self):
        """Timeout triggers fallback response."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return "Never reached"

        agent.generate = slow_generate

        config = AirlockConfig(generate_timeout=0.01, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test prompt")

        assert "timed out" in result.lower()
        assert proxy.metrics.timeout_errors == 1
        assert proxy.metrics.fallback_responses == 1

    @pytest.mark.asyncio
    async def test_timeout_updates_metrics(self):
        """Timeout updates metrics correctly."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)

        agent.generate = slow_generate

        config = AirlockConfig(generate_timeout=0.01, max_retries=0)
        proxy = AirlockProxy(agent, config)

        await proxy.generate("Test prompt")

        assert proxy.metrics.timeout_errors == 1
        assert proxy.metrics.fallback_responses == 1
        assert proxy.metrics.total_calls == 1
        assert proxy.metrics.successful_calls == 0

    @pytest.mark.asyncio
    async def test_timeout_raises_when_fallback_disabled(self):
        """Timeout raises exception when fallback is disabled."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)

        agent.generate = slow_generate

        config = AirlockConfig(
            generate_timeout=0.01,
            fallback_on_timeout=False,
            max_retries=0,
        )
        proxy = AirlockProxy(agent, config)

        with pytest.raises(asyncio.TimeoutError):
            await proxy.generate("Test prompt")


# =============================================================================
# Retry Handling Tests
# =============================================================================


class TestRetryHandling:
    """Test retry behavior."""

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        """Agent retries on timeout."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()
        call_count = 0

        async def flaky_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(10)  # First call times out
            return "Success on retry"

        agent.generate = flaky_generate

        config = AirlockConfig(
            generate_timeout=0.01,
            max_retries=1,
            retry_delay=0.001,  # Fast retry for test
        )
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test prompt")

        assert result == "Success on retry"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        """Agent retries on connection errors."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()
        call_count = 0

        async def flaky_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return "Success on retry"

        agent.generate = flaky_generate

        config = AirlockConfig(
            max_retries=1,
            retry_delay=0.001,
        )
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test prompt")

        assert result == "Success on retry"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_value_error(self):
        """Agent does not retry on ValueError (non-retryable)."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()
        call_count = 0

        async def bad_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        agent.generate = bad_generate

        config = AirlockConfig(
            max_retries=3,
            fallback_on_error=True,
        )
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test prompt")

        # Should not retry on ValueError
        assert call_count == 1
        assert "timed out" in result.lower() or "unable to generate" in result.lower()

    @pytest.mark.asyncio
    async def test_max_retries_respected(self):
        """Max retries limit is respected."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()
        call_count = 0

        async def always_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        agent.generate = always_fail

        config = AirlockConfig(
            max_retries=2,
            retry_delay=0.001,
            fallback_on_error=True,
        )
        proxy = AirlockProxy(agent, config)

        await proxy.generate("Test prompt")

        # Initial call + 2 retries = 3 total calls
        assert call_count == 3


# =============================================================================
# Response Sanitization Tests
# =============================================================================


class TestResponseSanitization:
    """Test response sanitization."""

    @pytest.mark.asyncio
    async def test_strips_markdown_code_fences(self):
        """Markdown code fences are stripped."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value='```json\n{"key": "value"}\n```')

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert "```" not in result
        assert '{"key": "value"}' in result

    @pytest.mark.asyncio
    async def test_strips_language_specific_fences(self):
        """Language-specific markdown fences are stripped."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="```python\nprint('hello')\n```")

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert "```python" not in result
        assert "```" not in result
        assert "print('hello')" in result

    @pytest.mark.asyncio
    async def test_removes_null_bytes(self):
        """Null bytes are removed."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="Hello\x00World")

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert "\x00" not in result
        assert "HelloWorld" in result

    @pytest.mark.asyncio
    async def test_removes_control_characters(self):
        """Control characters are removed."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="Hello\x07World\x1f")

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert "\x07" not in result
        assert "\x1f" not in result

    @pytest.mark.asyncio
    async def test_preserves_newlines_and_tabs(self):
        """Newlines and tabs are preserved."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="Line1\nLine2\tTabbed")

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert "\n" in result
        assert "\t" in result

    @pytest.mark.asyncio
    async def test_sanitization_updates_metrics(self):
        """Sanitization updates metrics."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="```json\n{}\n```")

        proxy = AirlockProxy(agent)
        await proxy.generate("Test prompt")

        assert proxy.metrics.sanitization_applied >= 1


class TestJsonExtraction:
    """Test JSON extraction from responses."""

    @pytest.mark.asyncio
    async def test_extracts_json_object_from_text(self):
        """JSON object is extracted from surrounding text."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(
            return_value='Here is the result: {"name": "test", "value": 42} Hope this helps!'
        )

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert result == '{"name": "test", "value": 42}'

    @pytest.mark.asyncio
    async def test_extracts_json_array_from_text(self):
        """JSON array is extracted from surrounding text."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="The list is: [1, 2, 3] as requested.")

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert result == "[1, 2, 3]"

    @pytest.mark.asyncio
    async def test_preserves_valid_json_at_start(self):
        """Valid JSON at start is preserved."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value='{"valid": "json"}')

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert result == '{"valid": "json"}'

    @pytest.mark.asyncio
    async def test_json_extraction_disabled(self):
        """JSON extraction can be disabled."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value='Here is the result: {"name": "test"}')

        config = AirlockConfig(extract_json=False)
        proxy = AirlockProxy(agent, config)
        result = await proxy.generate("Test prompt")

        # Should not extract JSON
        assert "Here is the result" in result


# =============================================================================
# Fallback Response Tests
# =============================================================================


class TestFallbackResponses:
    """Test fallback response generation."""

    @pytest.mark.asyncio
    async def test_generate_fallback_contains_agent_name(self):
        """Generate fallback contains agent name."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent(name="my_agent")

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)

        agent.generate = slow_generate

        config = AirlockConfig(generate_timeout=0.01, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test prompt")

        assert "my_agent" in result

    @pytest.mark.asyncio
    async def test_generate_fallback_contains_prompt_excerpt(self):
        """Generate fallback contains prompt excerpt."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)

        agent.generate = slow_generate

        config = AirlockConfig(generate_timeout=0.01, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Unique test prompt content")

        assert "Unique test prompt content" in result

    @pytest.mark.asyncio
    async def test_critique_fallback_returns_critique(self):
        """Critique fallback returns valid Critique-like dict."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy
        from aragora.core_types import Critique

        agent = MockAgent(name="my_agent")

        async def slow_critique(*args, **kwargs):
            await asyncio.sleep(10)

        agent.critique = slow_critique

        config = AirlockConfig(critique_timeout=0.01, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.critique("Proposal", "Task", target_agent="target")

        assert isinstance(result, Critique)
        assert result.agent == "my_agent"
        assert result.target_agent == "target"
        assert result.severity == 0.1  # Low severity for fallback

    @pytest.mark.asyncio
    async def test_vote_fallback_returns_vote(self):
        """Vote fallback returns valid Vote-like dict."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy
        from aragora.core_types import Vote

        agent = MockAgent(name="my_agent")

        async def slow_vote(*args, **kwargs):
            await asyncio.sleep(10)

        agent.vote = slow_vote

        config = AirlockConfig(vote_timeout=0.01, max_retries=0)
        proxy = AirlockProxy(agent, config)

        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        result = await proxy.vote(proposals, "Task")

        assert isinstance(result, Vote)
        assert result.agent == "my_agent"
        assert result.choice == "agent1"  # First proposal
        assert result.confidence == 0.1  # Low confidence for fallback


# =============================================================================
# wrap_agent() Tests
# =============================================================================


class TestWrapAgent:
    """Test wrap_agent convenience function."""

    def test_wraps_agent(self):
        """wrap_agent returns AirlockProxy."""
        from aragora.agents.airlock import AirlockProxy, wrap_agent

        agent = MockAgent()
        wrapped = wrap_agent(agent)

        assert isinstance(wrapped, AirlockProxy)
        assert wrapped.wrapped_agent is agent

    def test_wraps_with_config(self):
        """wrap_agent applies custom config."""
        from aragora.agents.airlock import AirlockConfig, wrap_agent

        agent = MockAgent()
        config = AirlockConfig(generate_timeout=30.0)
        wrapped = wrap_agent(agent, config)

        assert wrapped._config.generate_timeout == 30.0

    def test_does_not_double_wrap(self):
        """wrap_agent does not double-wrap AirlockProxy."""
        from aragora.agents.airlock import AirlockProxy, wrap_agent

        agent = MockAgent()
        wrapped_once = wrap_agent(agent)
        wrapped_twice = wrap_agent(wrapped_once)

        assert wrapped_twice is wrapped_once
        assert wrapped_twice.wrapped_agent is agent


# =============================================================================
# wrap_agents() Tests
# =============================================================================


class TestWrapAgents:
    """Test wrap_agents convenience function."""

    def test_wraps_multiple_agents(self):
        """wrap_agents wraps all agents."""
        from aragora.agents.airlock import AirlockProxy, wrap_agents

        agents = [MockAgent(name=f"agent_{i}") for i in range(3)]
        wrapped = wrap_agents(agents)

        assert len(wrapped) == 3
        for proxy in wrapped:
            assert isinstance(proxy, AirlockProxy)

    def test_wraps_with_shared_config(self):
        """wrap_agents applies shared config to all agents."""
        from aragora.agents.airlock import AirlockConfig, wrap_agents

        agents = [MockAgent(name=f"agent_{i}") for i in range(3)]
        config = AirlockConfig(generate_timeout=50.0)
        wrapped = wrap_agents(agents, config)

        for proxy in wrapped:
            assert proxy._config.generate_timeout == 50.0

    def test_does_not_double_wrap(self):
        """wrap_agents does not double-wrap already wrapped agents."""
        from aragora.agents.airlock import AirlockProxy, wrap_agents

        agents = [MockAgent(name="regular"), AirlockProxy(MockAgent(name="already_wrapped"))]
        wrapped = wrap_agents(agents)

        assert len(wrapped) == 2
        # First should be newly wrapped
        assert isinstance(wrapped[0], AirlockProxy)
        # Second should be same instance
        assert wrapped[1] is agents[1]

    def test_returns_list_of_airlock_proxies(self):
        """wrap_agents returns list of AirlockProxy."""
        from aragora.agents.airlock import AirlockProxy, wrap_agents

        agents = [MockAgent(name=f"agent_{i}") for i in range(2)]
        wrapped = wrap_agents(agents)

        assert isinstance(wrapped, list)
        assert all(isinstance(p, AirlockProxy) for p in wrapped)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_handles_os_error(self):
        """OSError is handled with retry and fallback."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()

        async def failing_generate(*args, **kwargs):
            raise OSError("File not found")

        agent.generate = failing_generate

        config = AirlockConfig(max_retries=1, retry_delay=0.001, fallback_on_error=True)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test prompt")

        assert "timed out" in result.lower() or "unable" in result.lower()
        assert proxy.metrics.fallback_responses >= 1

    @pytest.mark.asyncio
    async def test_handles_runtime_error(self):
        """RuntimeError is handled with retry and fallback."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()

        async def failing_generate(*args, **kwargs):
            raise RuntimeError("Unexpected error")

        agent.generate = failing_generate

        config = AirlockConfig(max_retries=1, retry_delay=0.001, fallback_on_error=True)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Test prompt")

        assert "timed out" in result.lower() or "unable" in result.lower()
        assert proxy.metrics.fallback_responses >= 1

    @pytest.mark.asyncio
    async def test_raises_when_fallback_disabled(self):
        """Error raises when fallback is disabled."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy

        agent = MockAgent()

        async def failing_generate(*args, **kwargs):
            raise ValueError("Bad input")

        agent.generate = failing_generate

        config = AirlockConfig(fallback_on_error=False)
        proxy = AirlockProxy(agent, config)

        with pytest.raises(ValueError, match="Bad input"):
            await proxy.generate("Test prompt")


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_response_is_sanitized(self):
        """Empty response is handled correctly."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="")

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        # Empty string should pass through (no sanitization needed)
        assert result == ""

    @pytest.mark.asyncio
    async def test_whitespace_only_response(self):
        """Whitespace-only response is stripped."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="   \n\t  ")

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_nested_json_extraction(self):
        """Nested JSON is extracted correctly."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value='Result: {"outer": {"inner": [1, 2, 3]}}')

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        assert '{"outer": {"inner": [1, 2, 3]}}' in result

    @pytest.mark.asyncio
    async def test_invalid_json_not_extracted(self):
        """Invalid JSON is not extracted."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        agent.generate = AsyncMock(return_value="Result: {invalid json here}")

        proxy = AirlockProxy(agent)
        result = await proxy.generate("Test prompt")

        # Should return original content since JSON is invalid
        assert "Result:" in result

    @pytest.mark.asyncio
    async def test_multiple_successful_calls_accumulate_latency(self):
        """Multiple successful calls accumulate latency."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent()
        proxy = AirlockProxy(agent)

        await proxy.generate("Prompt 1")
        await proxy.generate("Prompt 2")
        await proxy.generate("Prompt 3")

        assert proxy.metrics.total_calls == 3
        assert proxy.metrics.successful_calls == 3
        assert proxy.metrics.total_latency_ms > 0

    def test_agent_name_delegation_works(self):
        """Agent name is delegated correctly."""
        from aragora.agents.airlock import AirlockProxy

        agent = MockAgent(name="special_agent")
        proxy = AirlockProxy(agent)

        assert proxy.name == "special_agent"

    @pytest.mark.asyncio
    async def test_critique_with_target_agent(self):
        """Critique method passes target_agent to fallback."""
        from aragora.agents.airlock import AirlockConfig, AirlockProxy
        from aragora.core_types import Critique

        agent = MockAgent()

        async def slow_critique(*args, **kwargs):
            await asyncio.sleep(10)

        agent.critique = slow_critique

        config = AirlockConfig(critique_timeout=0.01, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.critique("Proposal", "Task", target_agent="specific_target")

        assert isinstance(result, Critique)
        assert result.target_agent == "specific_target"
