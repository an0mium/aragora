"""
Tests for AgentFallbackChain and related fallback utilities.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.agents.fallback import (
    AgentFallbackChain,
    AllProvidersExhaustedError,
    FallbackMetrics,
    QUOTA_ERROR_KEYWORDS,
)
from aragora.resilience import CircuitBreaker


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str = "mock", should_fail: bool = False, fail_message: str = "error"):
        self.name = name
        self.should_fail = should_fail
        self.fail_message = fail_message
        self.call_count = 0

    async def generate(self, prompt: str, context=None) -> str:
        self.call_count += 1
        if self.should_fail:
            raise Exception(self.fail_message)
        return f"Response from {self.name}"

    async def generate_stream(self, prompt: str, context=None):
        self.call_count += 1
        if self.should_fail:
            raise Exception(self.fail_message)
        for token in ["Hello", " ", "World"]:
            yield token


class TestFallbackMetrics:
    """Tests for FallbackMetrics."""

    def test_initial_state(self):
        """Metrics should start at zero."""
        metrics = FallbackMetrics()
        assert metrics.primary_attempts == 0
        assert metrics.primary_successes == 0
        assert metrics.fallback_attempts == 0
        assert metrics.fallback_successes == 0
        assert metrics.fallback_rate == 0.0
        assert metrics.success_rate == 0.0

    def test_record_primary_success(self):
        """Recording primary success should increment counters."""
        metrics = FallbackMetrics()
        metrics.record_primary_attempt(success=True)
        assert metrics.primary_attempts == 1
        assert metrics.primary_successes == 1
        assert metrics.total_failures == 0

    def test_record_primary_failure(self):
        """Recording primary failure should increment failure count."""
        metrics = FallbackMetrics()
        metrics.record_primary_attempt(success=False)
        assert metrics.primary_attempts == 1
        assert metrics.primary_successes == 0
        assert metrics.total_failures == 1

    def test_record_fallback_attempt(self):
        """Recording fallback attempt should track provider usage."""
        metrics = FallbackMetrics()
        metrics.record_fallback_attempt("openrouter", success=True)
        assert metrics.fallback_attempts == 1
        assert metrics.fallback_successes == 1
        assert metrics.fallback_providers_used == {"openrouter": 1}

    def test_fallback_rate_calculation(self):
        """Fallback rate should be calculated correctly."""
        metrics = FallbackMetrics()
        # 2 primary, 1 fallback = 33% fallback rate
        metrics.record_primary_attempt(success=True)
        metrics.record_primary_attempt(success=True)
        metrics.record_fallback_attempt("openrouter", success=True)
        assert metrics.fallback_rate == pytest.approx(1/3)

    def test_success_rate_calculation(self):
        """Success rate should include both primary and fallback."""
        metrics = FallbackMetrics()
        metrics.record_primary_attempt(success=True)
        metrics.record_primary_attempt(success=False)
        metrics.record_fallback_attempt("openrouter", success=True)
        # 2 successes out of 3 attempts
        assert metrics.success_rate == pytest.approx(2/3)


class TestAgentFallbackChain:
    """Tests for AgentFallbackChain."""

    @pytest.mark.asyncio
    async def test_primary_provider_success(self):
        """Should use primary provider when it succeeds."""
        chain = AgentFallbackChain(providers=["openai", "openrouter"])
        primary = MockAgent("openai")
        fallback = MockAgent("openrouter")

        chain.register_provider("openai", lambda: primary)
        chain.register_provider("openrouter", lambda: fallback)

        result = await chain.generate("test prompt")

        assert result == "Response from openai"
        assert primary.call_count == 1
        assert fallback.call_count == 0
        assert chain.metrics.primary_successes == 1
        assert chain.metrics.fallback_attempts == 0

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """Should fall back when primary provider fails."""
        chain = AgentFallbackChain(providers=["openai", "openrouter"])
        primary = MockAgent("openai", should_fail=True)
        fallback = MockAgent("openrouter")

        chain.register_provider("openai", lambda: primary)
        chain.register_provider("openrouter", lambda: fallback)

        result = await chain.generate("test prompt")

        assert result == "Response from openrouter"
        assert primary.call_count == 1
        assert fallback.call_count == 1
        assert chain.metrics.primary_attempts == 1
        assert chain.metrics.primary_successes == 0
        assert chain.metrics.fallback_successes == 1

    @pytest.mark.asyncio
    async def test_all_providers_exhausted(self):
        """Should raise AllProvidersExhaustedError when all fail."""
        chain = AgentFallbackChain(providers=["openai", "openrouter"])
        primary = MockAgent("openai", should_fail=True, fail_message="openai error")
        fallback = MockAgent("openrouter", should_fail=True, fail_message="openrouter error")

        chain.register_provider("openai", lambda: primary)
        chain.register_provider("openrouter", lambda: fallback)

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await chain.generate("test prompt")

        assert exc_info.value.providers == ["openai", "openrouter"]
        assert "openrouter error" in str(exc_info.value.last_error)

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Should skip circuit-broken providers."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
        chain = AgentFallbackChain(
            providers=["openai", "openrouter"],
            circuit_breaker=circuit_breaker,
        )

        primary = MockAgent("openai")
        fallback = MockAgent("openrouter")

        chain.register_provider("openai", lambda: primary)
        chain.register_provider("openrouter", lambda: fallback)

        # Trip the circuit breaker for openai
        circuit_breaker.record_failure("openai")
        circuit_breaker.record_failure("openai")

        result = await chain.generate("test prompt")

        # Should skip openai and use openrouter directly
        assert result == "Response from openrouter"
        assert primary.call_count == 0  # Skipped
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_records_failures(self):
        """Should record failures to circuit breaker."""
        circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
        chain = AgentFallbackChain(
            providers=["openai", "openrouter"],
            circuit_breaker=circuit_breaker,
        )

        primary = MockAgent("openai", should_fail=True)
        fallback = MockAgent("openrouter")

        chain.register_provider("openai", lambda: primary)
        chain.register_provider("openrouter", lambda: fallback)

        await chain.generate("test prompt")

        # openai should have 1 failure recorded
        assert circuit_breaker._failures.get("openai", 0) == 1
        # openrouter should have recorded success (resets failures)
        assert circuit_breaker._failures.get("openrouter", 0) == 0

    @pytest.mark.asyncio
    async def test_stream_primary_success(self):
        """Should stream from primary provider when it succeeds."""
        chain = AgentFallbackChain(providers=["openai", "openrouter"])
        primary = MockAgent("openai")
        fallback = MockAgent("openrouter")

        chain.register_provider("openai", lambda: primary)
        chain.register_provider("openrouter", lambda: fallback)

        tokens = []
        async for token in chain.generate_stream("test prompt"):
            tokens.append(token)

        assert tokens == ["Hello", " ", "World"]
        assert primary.call_count == 1
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_stream_fallback_on_failure(self):
        """Should fall back streaming when primary fails."""
        chain = AgentFallbackChain(providers=["openai", "openrouter"])
        primary = MockAgent("openai", should_fail=True)
        fallback = MockAgent("openrouter")

        chain.register_provider("openai", lambda: primary)
        chain.register_provider("openrouter", lambda: fallback)

        tokens = []
        async for token in chain.generate_stream("test prompt"):
            tokens.append(token)

        assert tokens == ["Hello", " ", "World"]
        assert primary.call_count == 1
        assert fallback.call_count == 1

    def test_get_available_providers(self):
        """Should return only non-circuit-broken providers."""
        circuit_breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
        chain = AgentFallbackChain(
            providers=["openai", "openrouter", "anthropic"],
            circuit_breaker=circuit_breaker,
        )

        # Trip openai
        circuit_breaker.record_failure("openai")
        circuit_breaker.record_failure("openai")

        available = chain.get_available_providers()
        assert available == ["openrouter", "anthropic"]

    def test_get_status(self):
        """Should return current chain status."""
        chain = AgentFallbackChain(providers=["openai", "openrouter"])
        chain.metrics.record_primary_attempt(success=True)
        chain.metrics.record_fallback_attempt("openrouter", success=True)

        status = chain.get_status()

        assert status["providers"] == ["openai", "openrouter"]
        assert status["available_providers"] == ["openai", "openrouter"]
        assert status["metrics"]["primary_attempts"] == 1
        assert status["metrics"]["fallback_attempts"] == 1
        assert "50.0%" in status["metrics"]["fallback_rate"]

    def test_reset_metrics(self):
        """Should reset all metrics."""
        chain = AgentFallbackChain(providers=["openai"])
        chain.metrics.record_primary_attempt(success=True)
        chain.metrics.record_fallback_attempt("openrouter", success=True)

        chain.reset_metrics()

        assert chain.metrics.primary_attempts == 0
        assert chain.metrics.fallback_attempts == 0


class TestRateLimitDetection:
    """Tests for rate limit error detection."""

    def test_detects_rate_limit_keywords(self):
        """Should detect rate limit errors from keywords."""
        chain = AgentFallbackChain(providers=["test"])

        assert chain._is_rate_limit_error(Exception("rate limit exceeded"))
        assert chain._is_rate_limit_error(Exception("quota exceeded"))
        assert chain._is_rate_limit_error(Exception("too many requests"))
        assert chain._is_rate_limit_error(Exception("Resource exhausted"))
        assert not chain._is_rate_limit_error(Exception("Unknown error"))

    def test_quota_error_keywords_coverage(self):
        """Should have comprehensive keyword coverage."""
        # Verify key keywords are present
        assert "rate limit" in QUOTA_ERROR_KEYWORDS
        assert "quota" in QUOTA_ERROR_KEYWORDS
        assert "too many requests" in QUOTA_ERROR_KEYWORDS
        assert "insufficient_quota" in QUOTA_ERROR_KEYWORDS


class TestAllProvidersExhaustedError:
    """Tests for AllProvidersExhaustedError."""

    def test_error_message(self):
        """Should include provider list and last error in message."""
        last_err = ValueError("API error")
        error = AllProvidersExhaustedError(["openai", "openrouter"], last_err)

        assert "openai" in str(error)
        assert "openrouter" in str(error)
        assert "API error" in str(error)
        assert error.providers == ["openai", "openrouter"]
        assert error.last_error is last_err

    def test_error_without_last_error(self):
        """Should work without last error."""
        error = AllProvidersExhaustedError(["openai"])

        assert "openai" in str(error)
        assert error.last_error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
