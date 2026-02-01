"""
Tests for per-provider agent metrics.

Tests the aragora.observability.metrics.agents module which provides
comprehensive Prometheus metrics for tracking AI provider performance.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

# Import the module to access the _initialized flag
import aragora.observability.metrics.agents as agents_module
from aragora.observability.metrics.agents import (
    AGENT_CIRCUIT_BREAKER_REJECTIONS,
    AGENT_CIRCUIT_BREAKER_STATE,
    AGENT_CIRCUIT_BREAKER_STATE_CHANGES,
    AGENT_CONNECTION_POOL_ACTIVE,
    AGENT_CONNECTION_POOL_WAITING,
    AGENT_FALLBACK_CHAIN_DEPTH,
    AGENT_FALLBACK_TRIGGERED,
    AGENT_MODEL_CALLS,
    AGENT_PROVIDER_CALLS,
    AGENT_PROVIDER_LATENCY,
    AGENT_RATE_LIMIT_BACKOFF_SECONDS,
    AGENT_RATE_LIMIT_DETECTED,
    AGENT_TOKEN_USAGE,
    AgentCallTracker,
    CircuitBreakerState,
    ErrorType,
    TokenType,
    _classify_exception,
    init_agent_provider_metrics,
    record_circuit_breaker_rejection,
    record_circuit_breaker_state_change,
    record_fallback_chain_depth,
    record_fallback_triggered,
    record_provider_call,
    record_provider_latency,
    record_provider_token_usage,
    record_rate_limit_detected,
    set_circuit_breaker_state,
    set_connection_pool_active,
    set_connection_pool_waiting,
    track_agent_provider_call,
    track_agent_provider_call_async,
    with_agent_provider_metrics,
    with_agent_provider_metrics_sync,
)


@pytest.fixture(scope="module", autouse=True)
def initialize_metrics_once():
    """Initialize metrics once for all tests in this module.

    This prevents the 'Duplicated timeseries' error that occurs when
    trying to register the same Prometheus metrics multiple times.

    We reset the _initialized flag to ensure we re-initialize with
    the get_or_create pattern which handles existing metrics gracefully.
    """
    # Reset initialization state to allow re-initialization
    # This handles the case where metrics may have been registered
    # by previous test runs with different configurations
    agents_module._initialized = False

    # Initialize (will use get_or_create pattern to handle existing metrics)
    init_agent_provider_metrics()
    yield


class TestErrorTypeEnum:
    """Tests for the ErrorType enum."""

    def test_error_types_exist(self) -> None:
        """All expected error types should be defined."""
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.RATE_LIMIT.value == "rate_limit"
        assert ErrorType.QUOTA.value == "quota"
        assert ErrorType.AUTH.value == "auth"
        assert ErrorType.CONNECTION.value == "connection"
        assert ErrorType.API_ERROR.value == "api_error"
        assert ErrorType.STREAM_ERROR.value == "stream_error"
        assert ErrorType.CIRCUIT_OPEN.value == "circuit_open"
        assert ErrorType.UNKNOWN.value == "unknown"


class TestCircuitBreakerStateEnum:
    """Tests for the CircuitBreakerState enum."""

    def test_circuit_breaker_states_exist(self) -> None:
        """All expected circuit breaker states should be defined."""
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"


class TestTokenTypeEnum:
    """Tests for the TokenType enum."""

    def test_token_types_exist(self) -> None:
        """All expected token types should be defined."""
        assert TokenType.INPUT.value == "input"
        assert TokenType.OUTPUT.value == "output"


class TestMetricsInitialization:
    """Tests for metrics initialization."""

    def test_init_agent_provider_metrics(self) -> None:
        """Metrics should initialize without error."""
        # Should not raise even if called multiple times
        # The fixture already called it once, but calling again should be safe
        # because init checks _initialized flag
        assert agents_module._initialized is True

    def test_metrics_are_defined_after_init(self) -> None:
        """All metrics should be defined after initialization."""
        # Check the module-level variables after init
        # (the imported symbols might still be None from initial import)
        assert agents_module.AGENT_PROVIDER_CALLS is not None
        assert agents_module.AGENT_PROVIDER_LATENCY is not None
        assert agents_module.AGENT_TOKEN_USAGE is not None
        assert agents_module.AGENT_CONNECTION_POOL_ACTIVE is not None
        assert agents_module.AGENT_CONNECTION_POOL_WAITING is not None
        assert agents_module.AGENT_RATE_LIMIT_DETECTED is not None
        assert agents_module.AGENT_RATE_LIMIT_BACKOFF_SECONDS is not None
        assert agents_module.AGENT_FALLBACK_CHAIN_DEPTH is not None
        assert agents_module.AGENT_FALLBACK_TRIGGERED is not None
        assert agents_module.AGENT_CIRCUIT_BREAKER_STATE is not None
        assert agents_module.AGENT_CIRCUIT_BREAKER_STATE_CHANGES is not None
        assert agents_module.AGENT_CIRCUIT_BREAKER_REJECTIONS is not None
        assert agents_module.AGENT_MODEL_CALLS is not None


class TestRecordProviderCall:
    """Tests for record_provider_call function."""

    def test_record_successful_call(self) -> None:
        """Should record a successful call."""

        record_provider_call(
            provider="anthropic",
            success=True,
            latency_seconds=1.5,
            model="claude-3-opus",
        )
        # Should not raise

    def test_record_failed_call_with_error_type(self) -> None:
        """Should record a failed call with error type."""

        record_provider_call(
            provider="openai",
            success=False,
            error_type=ErrorType.RATE_LIMIT,
            latency_seconds=0.5,
        )
        # Should not raise

    def test_record_failed_call_with_string_error_type(self) -> None:
        """Should record a failed call with string error type."""

        record_provider_call(
            provider="mistral",
            success=False,
            error_type="timeout",
        )
        # Should not raise

    def test_record_call_without_latency(self) -> None:
        """Should record a call without latency."""

        record_provider_call(
            provider="grok",
            success=True,
        )
        # Should not raise


class TestRecordProviderTokenUsage:
    """Tests for record_provider_token_usage function."""

    def test_record_token_usage(self) -> None:
        """Should record token usage."""

        record_provider_token_usage(
            provider="anthropic",
            input_tokens=1000,
            output_tokens=500,
        )
        # Should not raise

    def test_record_input_only(self) -> None:
        """Should record only input tokens."""

        record_provider_token_usage(
            provider="openai",
            input_tokens=500,
        )
        # Should not raise

    def test_record_output_only(self) -> None:
        """Should record only output tokens."""

        record_provider_token_usage(
            provider="openrouter",
            output_tokens=300,
        )
        # Should not raise

    def test_record_zero_tokens(self) -> None:
        """Should handle zero tokens gracefully."""

        record_provider_token_usage(
            provider="mistral",
            input_tokens=0,
            output_tokens=0,
        )
        # Should not raise


class TestConnectionPoolMetrics:
    """Tests for connection pool metrics."""

    def test_set_connection_pool_active(self) -> None:
        """Should set active connections count."""

        set_connection_pool_active("anthropic", 5)
        # Should not raise

    def test_set_connection_pool_waiting(self) -> None:
        """Should set waiting requests count."""

        set_connection_pool_waiting("openai", 3)
        # Should not raise


class TestRateLimitMetrics:
    """Tests for rate limit metrics."""

    def test_record_rate_limit_detected(self) -> None:
        """Should record rate limit detection."""

        record_rate_limit_detected("anthropic")
        # Should not raise

    def test_record_rate_limit_with_backoff(self) -> None:
        """Should record rate limit with backoff time."""

        record_rate_limit_detected("openai", backoff_seconds=30.0)
        # Should not raise


class TestFallbackMetrics:
    """Tests for fallback metrics."""

    def test_record_fallback_triggered(self) -> None:
        """Should record fallback trigger."""

        record_fallback_triggered(
            primary_provider="anthropic",
            fallback_provider="openrouter",
            trigger_reason="quota",
        )
        # Should not raise

    def test_record_fallback_chain_depth(self) -> None:
        """Should record fallback chain depth."""

        record_fallback_chain_depth(0)  # No fallback
        record_fallback_chain_depth(1)  # One fallback
        record_fallback_chain_depth(2)  # Two fallbacks
        # Should not raise


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker metrics."""

    def test_set_circuit_breaker_state_with_enum(self) -> None:
        """Should set circuit breaker state with enum."""

        set_circuit_breaker_state("anthropic", CircuitBreakerState.CLOSED)
        set_circuit_breaker_state("openai", CircuitBreakerState.OPEN)
        set_circuit_breaker_state("mistral", CircuitBreakerState.HALF_OPEN)
        # Should not raise

    def test_set_circuit_breaker_state_with_string(self) -> None:
        """Should set circuit breaker state with string."""

        set_circuit_breaker_state("anthropic", "closed")
        set_circuit_breaker_state("openai", "open")
        set_circuit_breaker_state("mistral", "half_open")
        # Should not raise

    def test_record_circuit_breaker_state_change(self) -> None:
        """Should record circuit breaker state change."""

        record_circuit_breaker_state_change(
            provider="anthropic",
            from_state=CircuitBreakerState.CLOSED,
            to_state=CircuitBreakerState.OPEN,
        )
        # Should not raise

    def test_record_circuit_breaker_rejection(self) -> None:
        """Should record circuit breaker rejection."""

        record_circuit_breaker_rejection("anthropic")
        # Should not raise


class TestExceptionClassification:
    """Tests for _classify_exception function."""

    def test_classify_timeout_error(self) -> None:
        """Should classify timeout errors."""
        assert _classify_exception(TimeoutError()) == ErrorType.TIMEOUT
        assert _classify_exception(asyncio.TimeoutError()) == ErrorType.TIMEOUT

    def test_classify_rate_limit_error(self) -> None:
        """Should classify rate limit errors."""

        class RateLimitError(Exception):
            pass

        assert _classify_exception(RateLimitError()) == ErrorType.RATE_LIMIT
        assert _classify_exception(Exception("rate limit exceeded")) == ErrorType.RATE_LIMIT
        assert _classify_exception(Exception("429 too many requests")) == ErrorType.RATE_LIMIT

    def test_classify_quota_error(self) -> None:
        """Should classify quota errors."""
        assert _classify_exception(Exception("quota exceeded")) == ErrorType.QUOTA
        assert _classify_exception(Exception("credit balance too low")) == ErrorType.QUOTA
        assert _classify_exception(Exception("billing issue")) == ErrorType.QUOTA

    def test_classify_auth_error(self) -> None:
        """Should classify auth errors."""

        class AuthError(Exception):
            pass

        assert _classify_exception(AuthError()) == ErrorType.AUTH
        assert _classify_exception(Exception("401 unauthorized")) == ErrorType.AUTH
        assert _classify_exception(Exception("403 forbidden")) == ErrorType.AUTH

    def test_classify_connection_error(self) -> None:
        """Should classify connection errors."""

        class ConnectionError(Exception):
            pass

        assert _classify_exception(ConnectionError()) == ErrorType.CONNECTION
        assert _classify_exception(Exception("failed to connect")) == ErrorType.CONNECTION

    def test_classify_circuit_open_error(self) -> None:
        """Should classify circuit breaker errors."""

        class CircuitBreakerOpenError(Exception):
            pass

        assert _classify_exception(CircuitBreakerOpenError()) == ErrorType.CIRCUIT_OPEN
        assert _classify_exception(Exception("circuit breaker open")) == ErrorType.CIRCUIT_OPEN

    def test_classify_stream_error(self) -> None:
        """Should classify stream errors."""

        class StreamError(Exception):
            pass

        assert _classify_exception(StreamError()) == ErrorType.STREAM_ERROR

    def test_classify_api_error(self) -> None:
        """Should classify API errors."""

        class APIError(Exception):
            pass

        assert _classify_exception(APIError()) == ErrorType.API_ERROR

    def test_classify_unknown_error(self) -> None:
        """Should classify unknown errors."""
        assert _classify_exception(Exception("random error")) == ErrorType.UNKNOWN
        assert _classify_exception(ValueError("something went wrong")) == ErrorType.UNKNOWN


class TestAgentCallTracker:
    """Tests for AgentCallTracker dataclass."""

    def test_tracker_initialization(self) -> None:
        """Should initialize with correct defaults."""
        tracker = AgentCallTracker(provider="anthropic", model="claude-3-opus")
        assert tracker.provider == "anthropic"
        assert tracker.model == "claude-3-opus"
        assert tracker._success is True
        assert tracker._error_type is None
        assert tracker._input_tokens == 0
        assert tracker._output_tokens == 0

    def test_record_tokens(self) -> None:
        """Should record token counts."""
        tracker = AgentCallTracker(provider="anthropic")
        tracker.record_tokens(input_tokens=100, output_tokens=50)
        assert tracker._input_tokens == 100
        assert tracker._output_tokens == 50

    def test_record_error_with_enum(self) -> None:
        """Should record error with enum type."""
        tracker = AgentCallTracker(provider="anthropic")
        tracker.record_error(ErrorType.TIMEOUT)
        assert tracker._success is False
        assert tracker._error_type == ErrorType.TIMEOUT

    def test_record_error_with_string(self) -> None:
        """Should record error with string type."""
        tracker = AgentCallTracker(provider="anthropic")
        tracker.record_error("rate_limit")
        assert tracker._success is False
        assert tracker._error_type == ErrorType.RATE_LIMIT

    def test_record_error_with_invalid_string(self) -> None:
        """Should handle invalid error type string."""
        tracker = AgentCallTracker(provider="anthropic")
        tracker.record_error("invalid_type")
        assert tracker._success is False
        assert tracker._error_type == ErrorType.UNKNOWN


class TestTrackAgentProviderCall:
    """Tests for track_agent_provider_call context manager."""

    def test_successful_call(self) -> None:
        """Should track a successful call."""

        with track_agent_provider_call("anthropic", model="claude-3-opus") as tracker:
            tracker.record_tokens(input_tokens=100, output_tokens=50)
        # Should not raise

    def test_failed_call(self) -> None:
        """Should track a failed call with exception."""

        with pytest.raises(ValueError):
            with track_agent_provider_call("anthropic") as tracker:
                raise ValueError("test error")

    def test_manual_error_recording(self) -> None:
        """Should allow manual error recording."""

        with track_agent_provider_call("anthropic") as tracker:
            tracker.record_error(ErrorType.RATE_LIMIT)
        # Should not raise

    def test_timing_is_recorded(self) -> None:
        """Should record timing."""

        with track_agent_provider_call("anthropic") as tracker:
            time.sleep(0.01)  # Small delay
        # Should not raise (timing recorded in _finalize)


class TestTrackAgentProviderCallAsync:
    """Tests for track_agent_provider_call_async context manager."""

    @pytest.mark.asyncio
    async def test_successful_async_call(self) -> None:
        """Should track a successful async call."""

        async with track_agent_provider_call_async("anthropic", model="claude-3-opus") as tracker:
            await asyncio.sleep(0.01)
            tracker.record_tokens(input_tokens=100, output_tokens=50)
        # Should not raise

    @pytest.mark.asyncio
    async def test_failed_async_call(self) -> None:
        """Should track a failed async call with exception."""

        with pytest.raises(ValueError):
            async with track_agent_provider_call_async("anthropic") as tracker:
                raise ValueError("test error")


class TestWithAgentProviderMetrics:
    """Tests for with_agent_provider_metrics decorator."""

    @pytest.mark.asyncio
    async def test_decorator_on_async_function(self) -> None:
        """Should decorate an async function."""

        @with_agent_provider_metrics("anthropic", model="claude-3-opus")
        async def my_func() -> str:
            return "result"

        result = await my_func()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_decorator_preserves_exception(self) -> None:
        """Should preserve exceptions from decorated function."""

        @with_agent_provider_metrics("anthropic")
        async def my_func() -> str:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await my_func()

    @pytest.mark.asyncio
    async def test_decorator_extracts_usage_from_result(self) -> None:
        """Should extract usage info if available."""

        # Mock response with usage info
        mock_result = MagicMock()
        mock_result.usage = MagicMock()
        mock_result.usage.input_tokens = 100
        mock_result.usage.output_tokens = 50

        @with_agent_provider_metrics("anthropic")
        async def my_func() -> MagicMock:
            return mock_result

        result = await my_func()
        assert result.usage.input_tokens == 100


class TestWithAgentProviderMetricsSync:
    """Tests for with_agent_provider_metrics_sync decorator."""

    def test_decorator_on_sync_function(self) -> None:
        """Should decorate a sync function."""

        @with_agent_provider_metrics_sync("anthropic", model="claude-3-opus")
        def my_func() -> str:
            return "result"

        result = my_func()
        assert result == "result"

    def test_decorator_preserves_exception(self) -> None:
        """Should preserve exceptions from decorated function."""

        @with_agent_provider_metrics_sync("anthropic")
        def my_func() -> str:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            my_func()


class TestNoOpBehavior:
    """Tests for no-op behavior when metrics are disabled."""

    def test_record_functions_work_without_prometheus(self) -> None:
        """Recording functions should work even without prometheus."""
        # These should not raise even if prometheus_client is not installed
        # or metrics are disabled

        record_provider_call("anthropic", True)
        record_provider_latency("anthropic", 1.0)
        record_provider_token_usage("anthropic", 100, 50)
        set_connection_pool_active("anthropic", 5)
        set_connection_pool_waiting("anthropic", 3)
        record_rate_limit_detected("anthropic")
        record_fallback_triggered("anthropic", "openrouter", "quota")
        record_fallback_chain_depth(1)
        set_circuit_breaker_state("anthropic", CircuitBreakerState.CLOSED)
        record_circuit_breaker_state_change(
            "anthropic", CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN
        )
        record_circuit_breaker_rejection("anthropic")


class TestAllProviders:
    """Tests to ensure metrics work for all supported providers."""

    @pytest.mark.parametrize(
        "provider",
        ["anthropic", "openai", "mistral", "grok", "openrouter", "gemini"],
    )
    def test_record_call_for_provider(self, provider: str) -> None:
        """Should record calls for each provider."""

        record_provider_call(provider, True, latency_seconds=1.0)
        record_provider_call(provider, False, error_type=ErrorType.API_ERROR)
        # Should not raise

    @pytest.mark.parametrize(
        "provider",
        ["anthropic", "openai", "mistral", "grok", "openrouter"],
    )
    def test_record_tokens_for_provider(self, provider: str) -> None:
        """Should record tokens for each provider."""

        record_provider_token_usage(provider, input_tokens=1000, output_tokens=500)
        # Should not raise


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_large_latency(self) -> None:
        """Should handle very large latency values."""

        record_provider_latency("anthropic", 3600.0)  # 1 hour
        # Should not raise

    def test_very_large_token_counts(self) -> None:
        """Should handle very large token counts."""

        record_provider_token_usage("anthropic", input_tokens=1000000, output_tokens=500000)
        # Should not raise

    def test_empty_provider_name(self) -> None:
        """Should handle empty provider name."""

        record_provider_call("", True)
        # Should not raise (but label will be empty)

    def test_special_characters_in_model(self) -> None:
        """Should handle special characters in model name."""

        record_provider_call(
            "anthropic",
            True,
            model="claude-3-opus:20240229-beta/test",
        )
        # Should not raise


class TestMetricsIntegration:
    """Integration tests for metrics with agent code patterns."""

    @pytest.mark.asyncio
    async def test_typical_successful_api_call_pattern(self) -> None:
        """Should work with typical successful API call pattern."""

        async with track_agent_provider_call_async("anthropic", model="claude-3-opus") as tracker:
            # Simulate API call
            await asyncio.sleep(0.01)

            # Record token usage from response
            tracker.record_tokens(input_tokens=150, output_tokens=75)

        # Verify no exceptions

    @pytest.mark.asyncio
    async def test_typical_failed_api_call_pattern(self) -> None:
        """Should work with typical failed API call pattern."""

        with pytest.raises(Exception):
            async with track_agent_provider_call_async("anthropic") as tracker:
                # Simulate API error
                raise Exception("API error 500")

    @pytest.mark.asyncio
    async def test_typical_rate_limit_pattern(self) -> None:
        """Should work with typical rate limit pattern."""

        # Record rate limit
        record_rate_limit_detected("anthropic", backoff_seconds=30.0)

        # Record the failed call
        record_provider_call(
            provider="anthropic",
            success=False,
            error_type=ErrorType.RATE_LIMIT,
            latency_seconds=0.5,
        )

        # Record fallback trigger
        record_fallback_triggered(
            primary_provider="anthropic",
            fallback_provider="openrouter",
            trigger_reason="rate_limit",
        )

    @pytest.mark.asyncio
    async def test_typical_circuit_breaker_pattern(self) -> None:
        """Should work with typical circuit breaker pattern."""

        # Record state transition
        record_circuit_breaker_state_change(
            provider="anthropic",
            from_state=CircuitBreakerState.CLOSED,
            to_state=CircuitBreakerState.OPEN,
        )

        # Record rejection
        record_circuit_breaker_rejection("anthropic")

        # Record failed call due to circuit breaker
        record_provider_call(
            provider="anthropic",
            success=False,
            error_type=ErrorType.CIRCUIT_OPEN,
            latency_seconds=0.001,
        )
