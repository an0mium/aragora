"""
Unit tests for APIAgent base class.

Tests the common functionality shared by all API-based agents:
- Token usage tracking (billing)
- Generation parameters (temperature, top_p, frequency_penalty)
- Circuit breaker integration via global registry
- Context prompt building
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from aragora.agents.api_agents.base import APIAgent
from aragora.core import Message
from aragora.resilience import CircuitBreaker, get_circuit_breaker, reset_all_circuit_breakers


# =============================================================================
# Concrete Test Implementation
# =============================================================================


class ConcreteAPIAgent(APIAgent):
    """Concrete implementation for testing the abstract base class."""

    async def generate(self, prompt: str, context=None):
        return "test response"

    async def critique(self, proposal: str, task: str, context=None):
        return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def api_agent():
    """Create a basic API agent for testing."""
    return ConcreteAPIAgent(name="test_agent", model="test-model")


@pytest.fixture
def api_agent_with_params():
    """Create an API agent with generation parameters."""
    return ConcreteAPIAgent(
        name="test_agent",
        model="test-model",
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=0.5,
    )


# =============================================================================
# Token Usage Tracking Tests
# =============================================================================


class TestTokenUsageTracking:
    """Tests for token usage tracking (billing support)."""

    def test_initial_token_counts_are_zero(self, api_agent):
        """Token counts start at zero."""
        assert api_agent.last_tokens_in == 0
        assert api_agent.last_tokens_out == 0
        assert api_agent.total_tokens_in == 0
        assert api_agent.total_tokens_out == 0

    def test_record_token_usage_updates_last(self, api_agent):
        """Recording usage updates last token counts."""
        api_agent._record_token_usage(tokens_in=100, tokens_out=50)

        assert api_agent.last_tokens_in == 100
        assert api_agent.last_tokens_out == 50

    def test_record_token_usage_accumulates_total(self, api_agent):
        """Recording usage accumulates total token counts."""
        api_agent._record_token_usage(tokens_in=100, tokens_out=50)
        api_agent._record_token_usage(tokens_in=200, tokens_out=75)

        assert api_agent.total_tokens_in == 300
        assert api_agent.total_tokens_out == 125

    def test_last_tokens_reset_on_new_call(self, api_agent):
        """Last tokens are overwritten on each call."""
        api_agent._record_token_usage(tokens_in=100, tokens_out=50)
        api_agent._record_token_usage(tokens_in=200, tokens_out=75)

        assert api_agent.last_tokens_in == 200
        assert api_agent.last_tokens_out == 75

    def test_get_token_usage_returns_dict(self, api_agent):
        """get_token_usage returns all token counts in a dict."""
        api_agent._record_token_usage(tokens_in=100, tokens_out=50)
        api_agent._record_token_usage(tokens_in=200, tokens_out=75)

        usage = api_agent.get_token_usage()

        assert usage["tokens_in"] == 200  # Last call
        assert usage["tokens_out"] == 75
        assert usage["total_tokens_in"] == 300
        assert usage["total_tokens_out"] == 125

    def test_reset_token_usage(self, api_agent):
        """reset_token_usage clears all counters."""
        api_agent._record_token_usage(tokens_in=100, tokens_out=50)
        api_agent._record_token_usage(tokens_in=200, tokens_out=75)

        api_agent.reset_token_usage()

        assert api_agent.last_tokens_in == 0
        assert api_agent.last_tokens_out == 0
        assert api_agent.total_tokens_in == 0
        assert api_agent.total_tokens_out == 0


# =============================================================================
# Generation Parameters Tests
# =============================================================================


class TestGenerationParameters:
    """Tests for generation parameter handling (persona diversity)."""

    def test_default_params_are_none(self, api_agent):
        """Generation params default to None (use provider defaults)."""
        assert api_agent.temperature is None
        assert api_agent.top_p is None
        assert api_agent.frequency_penalty is None

    def test_params_set_in_constructor(self, api_agent_with_params):
        """Generation params can be set in constructor."""
        assert api_agent_with_params.temperature == 0.7
        assert api_agent_with_params.top_p == 0.9
        assert api_agent_with_params.frequency_penalty == 0.5

    def test_set_generation_params(self, api_agent):
        """set_generation_params updates parameters."""
        api_agent.set_generation_params(
            temperature=0.8,
            top_p=0.95,
            frequency_penalty=0.3,
        )

        assert api_agent.temperature == 0.8
        assert api_agent.top_p == 0.95
        assert api_agent.frequency_penalty == 0.3

    def test_set_generation_params_partial(self, api_agent):
        """set_generation_params only updates provided params."""
        api_agent.temperature = 0.5

        api_agent.set_generation_params(top_p=0.8)

        assert api_agent.temperature == 0.5  # Unchanged
        assert api_agent.top_p == 0.8
        assert api_agent.frequency_penalty is None

    def test_get_generation_params_excludes_none(self, api_agent):
        """get_generation_params excludes None values."""
        api_agent.temperature = 0.7

        params = api_agent.get_generation_params()

        assert params == {"temperature": 0.7}
        assert "top_p" not in params
        assert "frequency_penalty" not in params

    def test_get_generation_params_all_set(self, api_agent_with_params):
        """get_generation_params includes all set values."""
        params = api_agent_with_params.get_generation_params()

        assert params == {
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
        }

    def test_get_generation_params_empty_when_none(self, api_agent):
        """get_generation_params returns empty dict when all None."""
        params = api_agent.get_generation_params()

        assert params == {}


# =============================================================================
# Circuit Breaker Integration Tests
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration via global registry."""

    def test_circuit_breaker_enabled_by_default(self, api_agent):
        """Circuit breaker is enabled by default."""
        assert api_agent.enable_circuit_breaker is True
        assert api_agent.circuit_breaker is not None

    def test_circuit_breaker_can_be_disabled(self):
        """Circuit breaker can be disabled."""
        agent = ConcreteAPIAgent(
            name="no_cb",
            model="test",
            enable_circuit_breaker=False,
        )

        assert agent.enable_circuit_breaker is False
        assert agent.circuit_breaker is None

    def test_circuit_breaker_can_be_injected(self):
        """Custom circuit breaker can be injected."""
        custom_cb = CircuitBreaker(failure_threshold=10, cooldown_seconds=120)

        agent = ConcreteAPIAgent(
            name="custom_cb",
            model="test",
            circuit_breaker=custom_cb,
        )

        assert agent.circuit_breaker is custom_cb

    def test_circuit_breaker_uses_global_registry(self):
        """Circuit breaker uses global registry for shared state."""
        reset_all_circuit_breakers()

        agent1 = ConcreteAPIAgent(name="shared", model="test")
        agent2 = ConcreteAPIAgent(name="shared", model="test")

        # Both agents should get the same circuit breaker instance
        assert agent1.circuit_breaker is agent2.circuit_breaker

    def test_circuit_breaker_custom_threshold(self):
        """Custom threshold and cooldown are respected."""
        reset_all_circuit_breakers()

        agent = ConcreteAPIAgent(
            name="custom_threshold",
            model="test",
            circuit_breaker_threshold=5,
            circuit_breaker_cooldown=30.0,
        )

        cb = agent.circuit_breaker
        assert cb is not None
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0

    def test_is_circuit_open_false_initially(self, api_agent):
        """Circuit is closed initially."""
        assert api_agent.is_circuit_open() is False

    def test_is_circuit_open_false_when_disabled(self):
        """is_circuit_open returns False when breaker disabled."""
        agent = ConcreteAPIAgent(
            name="disabled",
            model="test",
            enable_circuit_breaker=False,
        )

        assert agent.is_circuit_open() is False

    def test_circuit_opens_after_failures(self):
        """Circuit opens after threshold failures."""
        reset_all_circuit_breakers()

        agent = ConcreteAPIAgent(
            name="failing_agent",
            model="test",
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60.0,
        )

        cb = agent.circuit_breaker

        # Record failures to open circuit
        for _ in range(3):
            cb.record_failure()

        assert agent.is_circuit_open() is True


# =============================================================================
# Context Prompt Building Tests
# =============================================================================


class TestContextPromptBuilding:
    """Tests for _build_context_prompt method."""

    def test_build_context_prompt_empty(self, api_agent):
        """Empty context returns empty string."""
        result = api_agent._build_context_prompt(context=None)
        assert result == ""

    def test_build_context_prompt_single_message(self, api_agent):
        """Single message is formatted correctly."""
        context = [Message(role="proposer", agent="alice", content="Hello world")]

        result = api_agent._build_context_prompt(context=context)

        assert "alice" in result
        assert "Hello world" in result

    def test_build_context_prompt_multiple_messages(self, api_agent):
        """Multiple messages are all included."""
        context = [
            Message(role="proposer", agent="alice", content="First message"),
            Message(role="critic", agent="bob", content="Second message"),
            Message(role="proposer", agent="alice", content="Third message"),
        ]

        result = api_agent._build_context_prompt(context=context)

        assert "alice" in result
        assert "bob" in result
        assert "First message" in result
        assert "Second message" in result
        assert "Third message" in result

    def test_build_context_prompt_truncate_false_by_default(self, api_agent):
        """API agents don't truncate by default."""
        long_content = "x" * 10000
        context = [Message(role="proposer", agent="alice", content=long_content)]

        result = api_agent._build_context_prompt(context=context, truncate=False)

        # Full content should be present (not truncated)
        assert len(result) >= 10000


# =============================================================================
# Agent Type Tests
# =============================================================================


class TestAgentType:
    """Tests for agent_type attribute."""

    def test_default_agent_type(self, api_agent):
        """Default agent_type is 'api'."""
        assert api_agent.agent_type == "api"

    def test_agent_type_set_in_init(self):
        """Agent type is set during initialization."""
        agent = ConcreteAPIAgent(name="test", model="test-model")
        assert agent.agent_type == "api"


# =============================================================================
# Timeout Tests
# =============================================================================


class TestTimeout:
    """Tests for timeout configuration."""

    def test_default_timeout(self, api_agent):
        """Default timeout is 120 seconds."""
        assert api_agent.timeout == 120

    def test_custom_timeout(self):
        """Custom timeout is respected."""
        agent = ConcreteAPIAgent(name="test", model="test", timeout=60)
        assert agent.timeout == 60


# =============================================================================
# API Key Configuration Tests
# =============================================================================


class TestAPIKeyConfiguration:
    """Tests for API key and base URL configuration."""

    def test_api_key_default_none(self, api_agent):
        """API key defaults to None."""
        assert api_agent.api_key is None

    def test_api_key_can_be_set(self):
        """API key can be set in constructor."""
        agent = ConcreteAPIAgent(
            name="test",
            model="test",
            api_key="test-api-key",
        )
        assert agent.api_key == "test-api-key"

    def test_base_url_default_none(self, api_agent):
        """Base URL defaults to None."""
        assert api_agent.base_url is None

    def test_base_url_can_be_set(self):
        """Base URL can be set in constructor."""
        agent = ConcreteAPIAgent(
            name="test",
            model="test",
            base_url="https://custom.api.com",
        )
        assert agent.base_url == "https://custom.api.com"


# =============================================================================
# Role Configuration Tests
# =============================================================================


class TestRoleConfiguration:
    """Tests for agent role configuration."""

    def test_default_role_is_proposer(self, api_agent):
        """Default role is proposer."""
        assert api_agent.role == "proposer"

    def test_custom_role(self):
        """Custom role is respected."""
        agent = ConcreteAPIAgent(name="test", model="test", role="critic")
        assert agent.role == "critic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
