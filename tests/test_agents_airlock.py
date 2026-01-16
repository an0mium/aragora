"""
Tests for Agent Airlock resilience layer.

Tests cover:
- AirlockMetrics dataclass
- AirlockConfig dataclass
- AirlockProxy wrapper functionality
- Timeout handling and retries
- Response sanitization
- Fallback behavior
- Metrics collection
- Convenience functions
"""

from __future__ import annotations

import asyncio
import pytest
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, Mock, MagicMock, patch

from aragora.agents.airlock import (
    AirlockMetrics,
    AirlockConfig,
    AirlockProxy,
    wrap_agent,
    wrap_agents,
)
from aragora.core import Critique, Vote


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "mock-agent"
    model: str = "mock-model"

    async def generate(self, prompt: str, context: Optional[list] = None) -> str:
        return f"Response to: {prompt}"

    async def critique(self, proposal: str, task: str, context: Optional[list] = None) -> Critique:
        return Critique(
            agent=self.name,
            target_agent="other",
            target_content=proposal,
            issues=[],
            suggestions=[],
            severity=0.5,
            reasoning="Test critique",
        )

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        first_agent = next(iter(proposals.keys()))
        return Vote(
            agent=self.name,
            choice=first_agent,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=False,
        )


class SlowAgent(MockAgent):
    """Mock agent that takes a long time to respond."""

    def __init__(self, delay: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    async def generate(self, prompt: str, context: Optional[list] = None) -> str:
        await asyncio.sleep(self.delay)
        return f"Slow response to: {prompt}"

    async def critique(self, proposal: str, task: str, context: Optional[list] = None) -> Critique:
        await asyncio.sleep(self.delay)
        return await super().critique(proposal, task, context)

    async def vote(self, proposals: dict[str, str], task: str) -> Vote:
        await asyncio.sleep(self.delay)
        return await super().vote(proposals, task)


class FailingAgent(MockAgent):
    """Mock agent that raises exceptions."""

    def __init__(self, error: Exception = None, **kwargs):
        super().__init__(**kwargs)
        self.error = error or RuntimeError("Agent failed")
        self.call_count = 0

    async def generate(self, prompt: str, context: Optional[list] = None) -> str:
        self.call_count += 1
        raise self.error


class FlakeyAgent(MockAgent):
    """Mock agent that fails N times then succeeds."""

    def __init__(self, fail_count: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.fail_count = fail_count
        self.call_count = 0

    async def generate(self, prompt: str, context: Optional[list] = None) -> str:
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise RuntimeError(f"Failure {self.call_count}")
        return f"Success after {self.call_count} attempts"


# ============================================================================
# AirlockMetrics Tests
# ============================================================================


class TestAirlockMetrics:
    """Tests for AirlockMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = AirlockMetrics()
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.timeout_errors == 0
        assert metrics.sanitization_applied == 0
        assert metrics.fallback_responses == 0
        assert metrics.total_latency_ms == 0.0

    def test_success_rate_no_calls(self):
        """Test success rate with no calls returns 100%."""
        metrics = AirlockMetrics()
        assert metrics.success_rate == 100.0

    def test_success_rate_all_successful(self):
        """Test success rate with all successful calls."""
        metrics = AirlockMetrics(total_calls=10, successful_calls=10)
        assert metrics.success_rate == 100.0

    def test_success_rate_partial_success(self):
        """Test success rate with partial success."""
        metrics = AirlockMetrics(total_calls=10, successful_calls=7)
        assert metrics.success_rate == 70.0

    def test_avg_latency_no_calls(self):
        """Test average latency with no calls returns 0."""
        metrics = AirlockMetrics()
        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = AirlockMetrics(successful_calls=5, total_latency_ms=500.0)
        assert metrics.avg_latency_ms == 100.0

    def test_to_dict_format(self):
        """Test to_dict returns expected format."""
        metrics = AirlockMetrics(
            total_calls=10,
            successful_calls=8,
            timeout_errors=1,
            sanitization_applied=3,
            fallback_responses=2,
            total_latency_ms=800.0,
        )
        result = metrics.to_dict()

        assert result["total_calls"] == 10
        assert result["successful_calls"] == 8
        assert result["timeout_errors"] == 1
        assert result["sanitization_applied"] == 3
        assert result["fallback_responses"] == 2
        assert result["success_rate"] == 80.0
        assert result["avg_latency_ms"] == 100.0


# ============================================================================
# AirlockConfig Tests
# ============================================================================


class TestAirlockConfig:
    """Tests for AirlockConfig dataclass."""

    def test_default_timeouts(self):
        """Test default timeout values."""
        config = AirlockConfig()
        assert config.generate_timeout == 240.0
        assert config.critique_timeout == 180.0
        assert config.vote_timeout == 120.0

    def test_custom_timeouts(self):
        """Test custom timeout values."""
        config = AirlockConfig(generate_timeout=60.0, critique_timeout=45.0, vote_timeout=30.0)
        assert config.generate_timeout == 60.0
        assert config.critique_timeout == 45.0
        assert config.vote_timeout == 30.0

    def test_default_retry_settings(self):
        """Test default retry settings."""
        config = AirlockConfig()
        assert config.max_retries == 1
        assert config.retry_delay == 2.0

    def test_custom_retry_settings(self):
        """Test custom retry settings."""
        config = AirlockConfig(max_retries=3, retry_delay=1.0)
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

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


# ============================================================================
# AirlockProxy Basic Tests
# ============================================================================


class TestAirlockProxyBasic:
    """Basic tests for AirlockProxy class."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        assert proxy._agent is agent
        assert isinstance(proxy._config, AirlockConfig)
        assert isinstance(proxy._metrics, AirlockMetrics)

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        agent = MockAgent(name="test")
        config = AirlockConfig(generate_timeout=30.0)
        proxy = AirlockProxy(agent, config)

        assert proxy._config.generate_timeout == 30.0

    def test_metrics_property(self):
        """Test metrics property returns metrics."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        assert proxy.metrics is proxy._metrics

    def test_wrapped_agent_property(self):
        """Test wrapped_agent property returns agent."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        assert proxy.wrapped_agent is agent

    def test_getattr_delegation(self):
        """Test __getattr__ delegates to wrapped agent."""
        agent = MockAgent(name="test-agent", model="test-model")
        proxy = AirlockProxy(agent)

        assert proxy.name == "test-agent"
        assert proxy.model == "test-model"


# ============================================================================
# AirlockProxy Generate Tests
# ============================================================================


class TestAirlockProxyGenerate:
    """Tests for AirlockProxy.generate() method."""

    @pytest.mark.asyncio
    async def test_successful_generate(self):
        """Test successful generate call."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        result = await proxy.generate("Hello")

        assert "Response to: Hello" in result
        assert proxy.metrics.total_calls == 1
        assert proxy.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_generate_timeout_with_fallback(self):
        """Test generate timeout returns fallback."""
        agent = SlowAgent(name="slow", delay=10.0)
        config = AirlockConfig(generate_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Hello")

        assert "[Agent slow timed out" in result
        assert proxy.metrics.timeout_errors == 1
        assert proxy.metrics.fallback_responses == 1

    @pytest.mark.asyncio
    async def test_generate_timeout_raises_without_fallback(self):
        """Test generate timeout raises when fallback disabled."""
        agent = SlowAgent(name="slow", delay=10.0)
        config = AirlockConfig(generate_timeout=0.1, max_retries=0, fallback_on_timeout=False)
        proxy = AirlockProxy(agent, config)

        with pytest.raises(asyncio.TimeoutError):
            await proxy.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_error_with_fallback(self):
        """Test generate error returns fallback."""
        agent = FailingAgent(name="failing")
        config = AirlockConfig(max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Hello")

        assert "[Agent failing timed out" in result
        assert proxy.metrics.fallback_responses == 1

    @pytest.mark.asyncio
    async def test_generate_error_raises_without_fallback(self):
        """Test generate error raises when fallback disabled."""
        agent = FailingAgent(name="failing")
        config = AirlockConfig(max_retries=0, fallback_on_error=False)
        proxy = AirlockProxy(agent, config)

        with pytest.raises(RuntimeError, match="Agent failed"):
            await proxy.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_metrics_latency(self):
        """Test generate records latency metrics."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        await proxy.generate("Hello")

        assert proxy.metrics.total_latency_ms > 0


# ============================================================================
# AirlockProxy Retry Tests
# ============================================================================


class TestAirlockProxyRetry:
    """Tests for retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retries on transient failure."""
        agent = FlakeyAgent(name="flakey", fail_count=1)
        config = AirlockConfig(max_retries=2, retry_delay=0.01)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Hello")

        assert "Success after" in result
        assert agent.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test fallback when max retries exceeded."""
        agent = FlakeyAgent(name="flakey", fail_count=5)
        config = AirlockConfig(max_retries=2, retry_delay=0.01)
        proxy = AirlockProxy(agent, config)

        result = await proxy.generate("Hello")

        assert "[Agent flakey timed out" in result
        # 1 initial call + 2 retries = 3 total
        assert agent.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_delay_applied(self):
        """Test retry delay is applied between attempts."""
        agent = FlakeyAgent(name="flakey", fail_count=1)
        config = AirlockConfig(max_retries=1, retry_delay=0.1)
        proxy = AirlockProxy(agent, config)

        start = time.time()
        await proxy.generate("Hello")
        elapsed = time.time() - start

        # Should have waited at least retry_delay
        assert elapsed >= 0.1


# ============================================================================
# AirlockProxy Critique Tests
# ============================================================================


class TestAirlockProxyCritique:
    """Tests for AirlockProxy.critique() method."""

    @pytest.mark.asyncio
    async def test_successful_critique(self):
        """Test successful critique call."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        result = await proxy.critique("proposal", "task")

        assert result.agent == "test"
        assert proxy.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_critique_timeout_fallback(self):
        """Test critique timeout returns fallback Critique."""
        agent = SlowAgent(name="slow", delay=10.0)
        config = AirlockConfig(critique_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(agent, config)

        result = await proxy.critique("proposal", "task")

        assert result.agent == "slow"
        assert len(result.issues) > 0
        assert "unable to respond" in result.issues[0].lower()


# ============================================================================
# AirlockProxy Vote Tests
# ============================================================================


class TestAirlockProxyVote:
    """Tests for AirlockProxy.vote() method."""

    @pytest.mark.asyncio
    async def test_successful_vote(self):
        """Test successful vote call."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        proposals = {"agent1": "prop1", "agent2": "prop2"}
        result = await proxy.vote(proposals, "task")

        assert result.agent == "test"
        assert proxy.metrics.successful_calls == 1

    @pytest.mark.asyncio
    async def test_vote_timeout_fallback(self):
        """Test vote timeout returns fallback Vote."""
        agent = SlowAgent(name="slow", delay=10.0)
        config = AirlockConfig(vote_timeout=0.1, max_retries=0)
        proxy = AirlockProxy(agent, config)

        proposals = {"agent1": "prop1", "agent2": "prop2"}
        result = await proxy.vote(proposals, "task")

        assert result.agent == "slow"
        assert result.confidence == 0.1  # Low confidence fallback
        assert "Fallback" in result.reasoning or "fallback" in result.reasoning.lower()


# ============================================================================
# Response Sanitization Tests
# ============================================================================


class TestResponseSanitization:
    """Tests for response sanitization."""

    def test_strip_markdown_json_fence(self):
        """Test stripping ```json fences."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        content = '```json\n{"key": "value"}\n```'
        result = proxy._sanitize_response(content)

        assert result == '{"key": "value"}'

    def test_strip_markdown_python_fence(self):
        """Test stripping ```python fences."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        content = '```python\nprint("hello")\n```'
        result = proxy._sanitize_response(content)

        assert result == 'print("hello")'

    def test_remove_control_characters(self):
        """Test removing control characters."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        content = "Hello\x00World\x1fTest"
        result = proxy._sanitize_response(content)

        assert "\x00" not in result
        assert "\x1f" not in result
        assert "HelloWorldTest" in result

    def test_preserves_newlines_and_tabs(self):
        """Test preserves newlines and tabs."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        content = "Hello\nWorld\tTest"
        result = proxy._sanitize_response(content)

        assert "\n" in result
        assert "\t" in result

    def test_extract_json_from_text(self):
        """Test extracting JSON from surrounding text."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        content = 'Here is the JSON: {"key": "value"} as requested.'
        result = proxy._sanitize_response(content)

        assert result == '{"key": "value"}'

    def test_extract_json_array(self):
        """Test extracting JSON array from text."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        content = "The array is: [1, 2, 3] done."
        result = proxy._sanitize_response(content)

        assert result == "[1, 2, 3]"

    def test_invalid_json_not_extracted(self):
        """Test invalid JSON is not extracted."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        content = "Invalid: {not valid json}"
        result = proxy._sanitize_response(content)

        # Should return original (stripped) since JSON is invalid
        assert "Invalid" in result

    def test_sanitization_disabled(self):
        """Test sanitization can be disabled."""
        agent = MockAgent(name="test")
        config = AirlockConfig(extract_json=False, strip_markdown_fences=False)
        proxy = AirlockProxy(agent, config)

        content = '```json\n{"key": "value"}\n```'
        result = proxy._sanitize_response(content)

        assert "```json" in result

    def test_empty_content_handled(self):
        """Test empty content is handled."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        assert proxy._sanitize_response("") == ""
        assert proxy._sanitize_response(None) is None

    def test_sanitization_increments_metric(self):
        """Test sanitization increments metric counter."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        content = '```json\n{"key": "value"}\n```'
        proxy._sanitize_response(content)

        assert proxy.metrics.sanitization_applied == 1


# ============================================================================
# Fallback Generator Tests
# ============================================================================


class TestFallbackGenerators:
    """Tests for fallback response generators."""

    def test_generate_fallback_content(self):
        """Test generate fallback includes agent name and prompt."""
        agent = MockAgent(name="test-agent")
        proxy = AirlockProxy(agent)

        fallback = proxy._generate_fallback("Test prompt here")

        assert "test-agent" in fallback
        assert "Test prompt" in fallback
        assert "timed out" in fallback.lower()

    def test_generate_fallback_truncates_long_prompt(self):
        """Test generate fallback truncates long prompts."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        long_prompt = "x" * 200
        fallback = proxy._generate_fallback(long_prompt)

        # Should truncate to 100 chars
        assert len(fallback) < 200

    def test_critique_fallback_structure(self):
        """Test critique fallback has correct structure."""
        agent = MockAgent(name="critic")
        proxy = AirlockProxy(agent)

        fallback = proxy._critique_fallback("proposal", "task", "target_agent_name")

        assert fallback["agent"] == "critic"
        assert fallback["target_agent"] == "target_agent_name"
        assert "proposal" in fallback["target_content"]
        assert len(fallback["issues"]) > 0
        assert fallback["severity"] == 0.1

    def test_vote_fallback_structure(self):
        """Test vote fallback has correct structure."""
        agent = MockAgent(name="voter")
        proxy = AirlockProxy(agent)

        proposals = {"agent1": "prop1", "agent2": "prop2"}
        fallback = proxy._vote_fallback(proposals, "task")

        assert fallback["agent"] == "voter"
        assert fallback["choice"] == "agent1"  # First proposal
        assert fallback["confidence"] == 0.1
        assert fallback["continue_debate"] is False


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for wrap_agent and wrap_agents functions."""

    def test_wrap_agent_returns_proxy(self):
        """Test wrap_agent returns AirlockProxy."""
        agent = MockAgent(name="test")

        proxy = wrap_agent(agent)

        assert isinstance(proxy, AirlockProxy)
        assert proxy.wrapped_agent is agent

    def test_wrap_agent_with_config(self):
        """Test wrap_agent accepts custom config."""
        agent = MockAgent(name="test")
        config = AirlockConfig(generate_timeout=30.0)

        proxy = wrap_agent(agent, config)

        assert proxy._config.generate_timeout == 30.0

    def test_wrap_agents_returns_list(self):
        """Test wrap_agents returns list of proxies."""
        agents = [
            MockAgent(name="agent1"),
            MockAgent(name="agent2"),
            MockAgent(name="agent3"),
        ]

        proxies = wrap_agents(agents)

        assert len(proxies) == 3
        assert all(isinstance(p, AirlockProxy) for p in proxies)

    def test_wrap_agents_preserves_order(self):
        """Test wrap_agents preserves agent order."""
        agents = [
            MockAgent(name="first"),
            MockAgent(name="second"),
            MockAgent(name="third"),
        ]

        proxies = wrap_agents(agents)

        assert proxies[0].name == "first"
        assert proxies[1].name == "second"
        assert proxies[2].name == "third"

    def test_wrap_agents_with_shared_config(self):
        """Test wrap_agents applies same config to all."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        config = AirlockConfig(max_retries=5)

        proxies = wrap_agents(agents, config)

        assert proxies[0]._config.max_retries == 5
        assert proxies[1]._config.max_retries == 5

    def test_wrap_agents_empty_list(self):
        """Test wrap_agents handles empty list."""
        proxies = wrap_agents([])
        assert proxies == []


# ============================================================================
# Integration Tests
# ============================================================================


class TestAirlockIntegration:
    """Integration tests for airlock functionality."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow with generate, critique, vote."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        # Generate
        gen_result = await proxy.generate("What is 2+2?")
        assert "Response to" in gen_result

        # Critique
        critique_result = await proxy.critique(gen_result, "math task")
        assert critique_result.agent == "test"

        # Vote
        proposals = {"test": gen_result, "other": "4"}
        vote_result = await proxy.vote(proposals, "math task")
        assert vote_result.agent == "test"

        # Metrics should show 3 successful calls
        assert proxy.metrics.total_calls == 3
        assert proxy.metrics.successful_calls == 3

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        """Test proxy handles mixed success/failure."""
        # First agent succeeds, second times out
        agent1 = MockAgent(name="fast")
        agent2 = SlowAgent(name="slow", delay=10.0)

        config = AirlockConfig(generate_timeout=0.1, max_retries=0)
        proxies = wrap_agents([agent1, agent2], config)

        result1 = await proxies[0].generate("Hello")
        result2 = await proxies[1].generate("Hello")

        assert "Response to" in result1
        assert "[Agent slow timed out" in result2

    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test multiple concurrent calls work correctly."""
        agent = MockAgent(name="test")
        proxy = AirlockProxy(agent)

        tasks = [proxy.generate(f"Prompt {i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert proxy.metrics.total_calls == 5
        assert proxy.metrics.successful_calls == 5
