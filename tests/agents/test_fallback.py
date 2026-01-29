"""
Tests for quota detection and fallback utilities.

Tests the fallback module functionality including:
- QuotaFallbackMixin for quota error detection
- FallbackMetrics for tracking fallback chain behavior
- AgentFallbackChain for multi-provider sequencing
- Error handling (AllProvidersExhaustedError, FallbackTimeoutError)
- Local LLM provider detection
"""

from __future__ import annotations

import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# QUOTA_ERROR_KEYWORDS Tests
# =============================================================================


class TestQuotaErrorKeywords:
    """Test QUOTA_ERROR_KEYWORDS constant."""

    def test_keywords_is_frozenset(self):
        """Test keywords is a frozenset."""
        from aragora.agents.fallback import QUOTA_ERROR_KEYWORDS

        assert isinstance(QUOTA_ERROR_KEYWORDS, frozenset)

    def test_keywords_contains_rate_limit(self):
        """Test keywords contains rate limit variants."""
        from aragora.agents.fallback import QUOTA_ERROR_KEYWORDS

        assert "rate limit" in QUOTA_ERROR_KEYWORDS
        assert "rate_limit" in QUOTA_ERROR_KEYWORDS
        assert "too many requests" in QUOTA_ERROR_KEYWORDS

    def test_keywords_contains_quota(self):
        """Test keywords contains quota variants."""
        from aragora.agents.fallback import QUOTA_ERROR_KEYWORDS

        assert "quota" in QUOTA_ERROR_KEYWORDS
        assert "exceeded" in QUOTA_ERROR_KEYWORDS
        assert "resource exhausted" in QUOTA_ERROR_KEYWORDS

    def test_keywords_contains_billing(self):
        """Test keywords contains billing variants."""
        from aragora.agents.fallback import QUOTA_ERROR_KEYWORDS

        assert "billing" in QUOTA_ERROR_KEYWORDS
        assert "credit balance" in QUOTA_ERROR_KEYWORDS
        assert "insufficient" in QUOTA_ERROR_KEYWORDS


# =============================================================================
# QuotaFallbackMixin Tests
# =============================================================================


class MockAgentWithMixin:
    """Mock agent class using QuotaFallbackMixin."""

    OPENROUTER_MODEL_MAP = {
        "gpt-4o": "openai/gpt-4o",
        "claude-3-opus": "anthropic/claude-3-opus",
    }
    DEFAULT_FALLBACK_MODEL = "anthropic/claude-sonnet-4"

    def __init__(self, name: str = "test", model: str = "gpt-4o", enable_fallback: bool = True):
        self.name = name
        self.model = model
        self.enable_fallback = enable_fallback
        self.role = "proposer"
        self.timeout = 120
        self.system_prompt = None
        self._fallback_agent = None


class TestQuotaFallbackMixin:
    """Test QuotaFallbackMixin functionality."""

    def test_get_fallback_model_with_mapping(self):
        """Test getting fallback model with existing mapping."""
        from aragora.agents.fallback import QuotaFallbackMixin

        agent = MockAgentWithMixin(model="gpt-4o")
        # Add mixin method
        agent.get_fallback_model = QuotaFallbackMixin.get_fallback_model.__get__(
            agent, MockAgentWithMixin
        )

        result = agent.get_fallback_model()

        assert result == "openai/gpt-4o"

    def test_get_fallback_model_uses_default(self):
        """Test getting fallback model falls back to default."""
        from aragora.agents.fallback import QuotaFallbackMixin

        agent = MockAgentWithMixin(model="unknown-model")
        agent.get_fallback_model = QuotaFallbackMixin.get_fallback_model.__get__(
            agent, MockAgentWithMixin
        )

        result = agent.get_fallback_model()

        assert result == "anthropic/claude-sonnet-4"

    def test_is_quota_error_429(self):
        """Test 429 status is detected as quota error."""
        from aragora.agents.fallback import QuotaFallbackMixin

        agent = MockAgentWithMixin()
        agent.is_quota_error = QuotaFallbackMixin.is_quota_error.__get__(
            agent, MockAgentWithMixin
        )

        assert agent.is_quota_error(429, "") is True

    def test_is_quota_error_timeout_codes(self):
        """Test timeout status codes are detected."""
        from aragora.agents.fallback import QuotaFallbackMixin

        agent = MockAgentWithMixin()
        agent.is_quota_error = QuotaFallbackMixin.is_quota_error.__get__(
            agent, MockAgentWithMixin
        )

        assert agent.is_quota_error(408, "") is True  # Request Timeout
        assert agent.is_quota_error(504, "") is True  # Gateway Timeout
        assert agent.is_quota_error(524, "") is True  # Cloudflare timeout

    def test_is_quota_error_403_with_quota_keyword(self):
        """Test 403 with quota keyword is detected."""
        from aragora.agents.fallback import QuotaFallbackMixin

        agent = MockAgentWithMixin()
        agent.is_quota_error = QuotaFallbackMixin.is_quota_error.__get__(
            agent, MockAgentWithMixin
        )

        assert agent.is_quota_error(403, "Quota exceeded for this project") is True
        assert agent.is_quota_error(403, "Permission denied") is False

    def test_is_quota_error_400_with_billing_keyword(self):
        """Test 400 with billing keyword is detected."""
        from aragora.agents.fallback import QuotaFallbackMixin

        agent = MockAgentWithMixin()
        agent.is_quota_error = QuotaFallbackMixin.is_quota_error.__get__(
            agent, MockAgentWithMixin
        )

        assert agent.is_quota_error(400, "Credit balance is too low") is True
        assert agent.is_quota_error(400, "Invalid request") is False

    def test_is_quota_error_timeout_in_text(self):
        """Test timeout keyword in error text is detected."""
        from aragora.agents.fallback import QuotaFallbackMixin

        agent = MockAgentWithMixin()
        agent.is_quota_error = QuotaFallbackMixin.is_quota_error.__get__(
            agent, MockAgentWithMixin
        )

        assert agent.is_quota_error(500, "Request timed out") is True
        assert agent.is_quota_error(500, "Connection timeout") is True


# =============================================================================
# FallbackMetrics Tests
# =============================================================================


class TestFallbackMetrics:
    """Test FallbackMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initializes with zeros."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()

        assert metrics.primary_attempts == 0
        assert metrics.primary_successes == 0
        assert metrics.fallback_attempts == 0
        assert metrics.fallback_successes == 0
        assert metrics.total_failures == 0

    def test_record_primary_attempt_success(self):
        """Test recording successful primary attempt."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()
        metrics.record_primary_attempt(success=True)

        assert metrics.primary_attempts == 1
        assert metrics.primary_successes == 1
        assert metrics.total_failures == 0

    def test_record_primary_attempt_failure(self):
        """Test recording failed primary attempt."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()
        metrics.record_primary_attempt(success=False)

        assert metrics.primary_attempts == 1
        assert metrics.primary_successes == 0
        assert metrics.total_failures == 1

    def test_record_fallback_attempt_success(self):
        """Test recording successful fallback attempt."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()
        metrics.record_fallback_attempt("openrouter", success=True)

        assert metrics.fallback_attempts == 1
        assert metrics.fallback_successes == 1
        assert metrics.fallback_providers_used == {"openrouter": 1}

    def test_record_fallback_attempt_failure(self):
        """Test recording failed fallback attempt."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()
        metrics.record_fallback_attempt("openrouter", success=False)

        assert metrics.fallback_attempts == 1
        assert metrics.fallback_successes == 0
        assert metrics.total_failures == 1

    def test_fallback_rate_calculation(self):
        """Test fallback rate calculation."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()
        metrics.primary_attempts = 8
        metrics.fallback_attempts = 2

        assert metrics.fallback_rate == 0.2  # 2/10

    def test_fallback_rate_zero_attempts(self):
        """Test fallback rate with zero attempts."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()

        assert metrics.fallback_rate == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()
        metrics.primary_attempts = 10
        metrics.primary_successes = 7
        metrics.fallback_attempts = 3
        metrics.fallback_successes = 2

        # Total successes: 7 + 2 = 9
        # Total attempts: 10 + 3 = 13
        expected_rate = 9 / 13
        assert abs(metrics.success_rate - expected_rate) < 0.001

    def test_success_rate_zero_attempts(self):
        """Test success rate with zero attempts."""
        from aragora.agents.fallback import FallbackMetrics

        metrics = FallbackMetrics()

        assert metrics.success_rate == 0.0


# =============================================================================
# Exception Tests
# =============================================================================


class TestAllProvidersExhaustedError:
    """Test AllProvidersExhaustedError exception."""

    def test_error_with_providers(self):
        """Test error with provider list."""
        from aragora.agents.fallback import AllProvidersExhaustedError

        error = AllProvidersExhaustedError(["openai", "openrouter", "anthropic"])

        assert error.providers == ["openai", "openrouter", "anthropic"]
        assert "openai" in str(error)
        assert "openrouter" in str(error)

    def test_error_with_last_error(self):
        """Test error with last error."""
        from aragora.agents.fallback import AllProvidersExhaustedError

        last_error = RuntimeError("API error")
        error = AllProvidersExhaustedError(["openai"], last_error=last_error)

        assert error.last_error is last_error
        assert "API error" in str(error)


class TestFallbackTimeoutError:
    """Test FallbackTimeoutError exception."""

    def test_error_attributes(self):
        """Test error stores attributes correctly."""
        from aragora.agents.fallback import FallbackTimeoutError

        error = FallbackTimeoutError(
            elapsed=45.5, limit=30.0, tried=["openai", "openrouter"]
        )

        assert error.elapsed == 45.5
        assert error.limit == 30.0
        assert error.tried_providers == ["openai", "openrouter"]

    def test_error_message(self):
        """Test error message format."""
        from aragora.agents.fallback import FallbackTimeoutError

        error = FallbackTimeoutError(
            elapsed=45.5, limit=30.0, tried=["openai", "openrouter"]
        )

        assert "45.5s" in str(error)
        assert "30" in str(error)
        assert "openai" in str(error)


# =============================================================================
# AgentFallbackChain Tests
# =============================================================================


class TestAgentFallbackChainInit:
    """Test AgentFallbackChain initialization."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai", "openrouter"])

        assert chain.providers == ["openai", "openrouter"]
        assert chain.max_retries == AgentFallbackChain.DEFAULT_MAX_RETRIES
        assert chain.max_fallback_time == AgentFallbackChain.DEFAULT_MAX_FALLBACK_TIME

    def test_init_with_custom_limits(self):
        """Test initialization with custom limits."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(
            providers=["openai"],
            max_retries=3,
            max_fallback_time=60.0,
        )

        assert chain.max_retries == 3
        assert chain.max_fallback_time == 60.0

    def test_init_with_circuit_breaker(self):
        """Test initialization with circuit breaker."""
        from aragora.agents.fallback import AgentFallbackChain
        from aragora.resilience import CircuitBreaker

        cb = CircuitBreaker(name="test", failure_threshold=3)
        chain = AgentFallbackChain(providers=["openai"], circuit_breaker=cb)

        assert chain.circuit_breaker is cb


class TestAgentFallbackChainProviders:
    """Test AgentFallbackChain provider management."""

    def test_register_provider(self):
        """Test registering a provider factory."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])

        mock_factory = MagicMock(return_value=MagicMock())
        chain.register_provider("openai", mock_factory)

        assert "openai" in chain._provider_factories

    def test_get_agent_from_factory(self):
        """Test getting agent from registered factory."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])

        mock_agent = MagicMock()
        mock_factory = MagicMock(return_value=mock_agent)
        chain.register_provider("openai", mock_factory)

        result = chain._get_agent("openai")

        assert result is mock_agent
        mock_factory.assert_called_once()

    def test_get_agent_caches_result(self):
        """Test agent is cached after first creation."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])

        mock_agent = MagicMock()
        mock_factory = MagicMock(return_value=mock_agent)
        chain.register_provider("openai", mock_factory)

        # Get agent twice
        chain._get_agent("openai")
        result = chain._get_agent("openai")

        # Factory should only be called once
        assert mock_factory.call_count == 1
        assert result is mock_agent

    def test_get_agent_returns_instance_directly(self):
        """Test agent instance is returned directly."""
        from aragora.agents.fallback import AgentFallbackChain

        mock_agent = MagicMock()
        mock_agent.name = "direct-agent"

        chain = AgentFallbackChain(providers=[mock_agent])

        result = chain._get_agent(mock_agent)

        assert result is mock_agent

    def test_provider_key_for_string(self):
        """Test provider key for string provider."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])

        assert chain._provider_key("openai") == "openai"

    def test_provider_key_for_agent(self):
        """Test provider key for agent instance."""
        from aragora.agents.fallback import AgentFallbackChain

        mock_agent = MagicMock()
        mock_agent.name = "my-agent"

        chain = AgentFallbackChain(providers=[mock_agent])

        assert chain._provider_key(mock_agent) == "my-agent"


class TestAgentFallbackChainAvailability:
    """Test provider availability checking."""

    def test_get_available_providers_without_circuit_breaker(self):
        """Test all providers available without circuit breaker."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai", "openrouter"])

        available = chain.get_available_providers()

        assert available == ["openai", "openrouter"]

    def test_is_available_without_circuit_breaker(self):
        """Test provider is available without circuit breaker."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])

        assert chain._is_available("openai") is True


class TestAgentFallbackChainGenerate:
    """Test generate method."""

    @pytest.mark.asyncio
    async def test_generate_success_first_provider(self):
        """Test successful generation with first provider."""
        from aragora.agents.fallback import AgentFallbackChain

        mock_agent = MagicMock()
        mock_agent.name = "openai"
        mock_agent.generate = AsyncMock(return_value="Response from OpenAI")

        chain = AgentFallbackChain(providers=[mock_agent])

        result = await chain.generate("Test prompt")

        assert result == "Response from OpenAI"
        mock_agent.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_fallback_on_failure(self):
        """Test fallback when primary fails."""
        from aragora.agents.fallback import AgentFallbackChain

        primary = MagicMock()
        primary.name = "openai"
        primary.generate = AsyncMock(side_effect=RuntimeError("API error"))

        fallback = MagicMock()
        fallback.name = "openrouter"
        fallback.generate = AsyncMock(return_value="Response from fallback")

        chain = AgentFallbackChain(providers=[primary, fallback])

        result = await chain.generate("Test prompt")

        assert result == "Response from fallback"
        primary.generate.assert_called_once()
        fallback.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_all_providers_exhausted(self):
        """Test error when all providers fail."""
        from aragora.agents.fallback import AgentFallbackChain, AllProvidersExhaustedError

        agent1 = MagicMock()
        agent1.name = "openai"
        agent1.generate = AsyncMock(side_effect=RuntimeError("Error 1"))

        agent2 = MagicMock()
        agent2.name = "openrouter"
        agent2.generate = AsyncMock(side_effect=RuntimeError("Error 2"))

        chain = AgentFallbackChain(providers=[agent1, agent2])

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await chain.generate("Test prompt")

        assert "openai" in exc_info.value.providers
        assert "openrouter" in exc_info.value.providers

    @pytest.mark.asyncio
    async def test_generate_respects_max_retries(self):
        """Test generate respects max_retries limit."""
        from aragora.agents.fallback import AgentFallbackChain

        agents = []
        for i in range(5):
            agent = MagicMock()
            agent.name = f"agent-{i}"
            agent.generate = AsyncMock(side_effect=RuntimeError(f"Error {i}"))
            agents.append(agent)

        chain = AgentFallbackChain(providers=agents, max_retries=2)

        with pytest.raises(Exception):
            await chain.generate("Test prompt")

        # Only first 2 agents should be tried
        agents[0].generate.assert_called_once()
        agents[1].generate.assert_called_once()
        agents[2].generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_records_metrics(self):
        """Test generate records metrics correctly."""
        from aragora.agents.fallback import AgentFallbackChain

        primary = MagicMock()
        primary.name = "openai"
        primary.generate = AsyncMock(side_effect=RuntimeError("API error"))

        fallback = MagicMock()
        fallback.name = "openrouter"
        fallback.generate = AsyncMock(return_value="Response")

        chain = AgentFallbackChain(providers=[primary, fallback])

        await chain.generate("Test prompt")

        assert chain.metrics.primary_attempts == 1
        assert chain.metrics.primary_successes == 0
        assert chain.metrics.fallback_attempts == 1
        assert chain.metrics.fallback_successes == 1


class TestAgentFallbackChainStatus:
    """Test status and metrics methods."""

    def test_get_status(self):
        """Test get_status returns expected structure."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(
            providers=["openai", "openrouter"],
            max_retries=3,
            max_fallback_time=60.0,
        )

        status = chain.get_status()

        assert "providers" in status
        assert "available_providers" in status
        assert "limits" in status
        assert "metrics" in status
        assert status["limits"]["max_retries"] == 3
        assert status["limits"]["max_fallback_time"] == 60.0

    def test_reset_metrics(self):
        """Test reset_metrics clears all counters."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])
        chain.metrics.primary_attempts = 10
        chain.metrics.fallback_attempts = 5

        chain.reset_metrics()

        assert chain.metrics.primary_attempts == 0
        assert chain.metrics.fallback_attempts == 0


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestGetDefaultFallbackEnabled:
    """Test get_default_fallback_enabled function."""

    def test_returns_boolean(self):
        """Test function returns a boolean."""
        from aragora.agents.fallback import get_default_fallback_enabled

        result = get_default_fallback_enabled()

        assert isinstance(result, bool)


class TestGetLocalFallbackProviders:
    """Test get_local_fallback_providers function."""

    def test_returns_list(self):
        """Test function returns a list."""
        from aragora.agents.fallback import get_local_fallback_providers

        result = get_local_fallback_providers()

        assert isinstance(result, list)


class TestIsLocalLlmAvailable:
    """Test is_local_llm_available function."""

    def test_returns_boolean(self):
        """Test function returns a boolean."""
        from aragora.agents.fallback import is_local_llm_available

        result = is_local_llm_available()

        assert isinstance(result, bool)


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test module exports."""

    def test_quota_error_keywords_exportable(self):
        """Test QUOTA_ERROR_KEYWORDS can be imported."""
        from aragora.agents.fallback import QUOTA_ERROR_KEYWORDS

        assert QUOTA_ERROR_KEYWORDS is not None

    def test_mixin_exportable(self):
        """Test QuotaFallbackMixin can be imported."""
        from aragora.agents.fallback import QuotaFallbackMixin

        assert QuotaFallbackMixin is not None

    def test_metrics_exportable(self):
        """Test FallbackMetrics can be imported."""
        from aragora.agents.fallback import FallbackMetrics

        assert FallbackMetrics is not None

    def test_chain_exportable(self):
        """Test AgentFallbackChain can be imported."""
        from aragora.agents.fallback import AgentFallbackChain

        assert AgentFallbackChain is not None

    def test_errors_exportable(self):
        """Test error classes can be imported."""
        from aragora.agents.fallback import (
            AllProvidersExhaustedError,
            FallbackTimeoutError,
        )

        assert AllProvidersExhaustedError is not None
        assert FallbackTimeoutError is not None

    def test_utility_functions_exportable(self):
        """Test utility functions can be imported."""
        from aragora.agents.fallback import (
            build_fallback_chain_with_local,
            get_default_fallback_enabled,
            get_local_fallback_providers,
            is_local_llm_available,
        )

        assert get_default_fallback_enabled is not None
        assert get_local_fallback_providers is not None
        assert is_local_llm_available is not None
        assert build_fallback_chain_with_local is not None


# =============================================================================
# Rate Limit Detection Tests
# =============================================================================


class TestRateLimitDetection:
    """Test rate limit error detection in fallback chain."""

    def test_is_rate_limit_error_true(self):
        """Test rate limit detection returns true for rate limit errors."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])

        error = RuntimeError("Rate limit exceeded")
        assert chain._is_rate_limit_error(error) is True

    def test_is_rate_limit_error_false(self):
        """Test rate limit detection returns false for other errors."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])

        error = RuntimeError("Connection refused")
        assert chain._is_rate_limit_error(error) is False

    def test_is_rate_limit_error_quota(self):
        """Test rate limit detection for quota errors."""
        from aragora.agents.fallback import AgentFallbackChain

        chain = AgentFallbackChain(providers=["openai"])

        error = RuntimeError("Quota exceeded for this API key")
        assert chain._is_rate_limit_error(error) is True
