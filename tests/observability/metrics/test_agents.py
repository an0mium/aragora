"""Tests for observability/metrics/agents.py — agent provider metrics."""

import asyncio
from unittest.mock import patch

import pytest

from aragora.observability.metrics import agents as mod
from aragora.observability.metrics.agents import (
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


@pytest.fixture(autouse=True)
def _reset_module():
    """Reset module initialization state between tests."""
    mod._initialized = False
    yield
    mod._initialized = False


# =============================================================================
# Enums
# =============================================================================


class TestEnums:
    def test_circuit_breaker_states(self):
        assert CircuitBreakerState.OPEN.value == "open"
        assert CircuitBreakerState.CLOSED.value == "closed"
        assert CircuitBreakerState.HALF_OPEN.value == "half_open"

    def test_error_types(self):
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.RATE_LIMIT.value == "rate_limit"
        assert ErrorType.QUOTA.value == "quota"
        assert ErrorType.AUTH.value == "auth"
        assert ErrorType.CONNECTION.value == "connection"
        assert ErrorType.UNKNOWN.value == "unknown"

    def test_token_types(self):
        assert TokenType.INPUT.value == "input"
        assert TokenType.OUTPUT.value == "output"


# =============================================================================
# Initialization
# =============================================================================


class TestInitialization:
    def test_init_sets_noop_when_disabled(self):
        with patch("aragora.observability.metrics.agents.get_metrics_enabled", return_value=False):
            init_agent_provider_metrics()
            assert mod._initialized is True
            assert mod.AGENT_PROVIDER_CALLS is not None

    def test_init_idempotent(self):
        with patch("aragora.observability.metrics.agents.get_metrics_enabled", return_value=False):
            init_agent_provider_metrics()
            first = mod.AGENT_PROVIDER_CALLS
            init_agent_provider_metrics()
            assert mod.AGENT_PROVIDER_CALLS is first

    def test_ensure_init_triggers_init(self):
        with patch("aragora.observability.metrics.agents.get_metrics_enabled", return_value=False):
            mod._ensure_init()
            assert mod._initialized is True


# =============================================================================
# Recording Functions (NoOp mode)
# =============================================================================


class TestRecordingFunctions:
    @pytest.fixture(autouse=True)
    def _init_noop(self):
        with patch("aragora.observability.metrics.agents.get_metrics_enabled", return_value=False):
            init_agent_provider_metrics()

    def test_record_provider_call_success(self):
        record_provider_call("anthropic", success=True, latency_seconds=1.5)

    def test_record_provider_call_failure(self):
        record_provider_call(
            "openai", success=False, error_type=ErrorType.TIMEOUT, latency_seconds=30.0
        )

    def test_record_provider_call_error_string(self):
        record_provider_call("mistral", success=False, error_type="rate_limit")

    def test_record_provider_call_with_model(self):
        record_provider_call("anthropic", success=True, model="claude-3-opus")

    def test_record_provider_latency(self):
        record_provider_latency("openai", 2.5)

    def test_record_token_usage(self):
        record_provider_token_usage("anthropic", input_tokens=100, output_tokens=50)

    def test_record_token_usage_zero(self):
        record_provider_token_usage("anthropic", input_tokens=0, output_tokens=0)

    def test_set_connection_pool_active(self):
        set_connection_pool_active("openai", 5)

    def test_set_connection_pool_waiting(self):
        set_connection_pool_waiting("openai", 2)

    def test_record_rate_limit_detected(self):
        record_rate_limit_detected("anthropic", backoff_seconds=30.0)

    def test_record_rate_limit_no_backoff(self):
        record_rate_limit_detected("openai")

    def test_record_fallback_triggered(self):
        record_fallback_triggered("anthropic", "openrouter", "rate_limit")

    def test_record_fallback_chain_depth(self):
        record_fallback_chain_depth(2)

    def test_set_circuit_breaker_state_enum(self):
        set_circuit_breaker_state("anthropic", CircuitBreakerState.OPEN)

    def test_set_circuit_breaker_state_string(self):
        set_circuit_breaker_state("openai", "closed")

    def test_record_circuit_breaker_state_change(self):
        record_circuit_breaker_state_change(
            "anthropic", CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN
        )

    def test_record_circuit_breaker_state_change_strings(self):
        record_circuit_breaker_state_change("openai", "closed", "half_open")

    def test_record_circuit_breaker_rejection(self):
        record_circuit_breaker_rejection("anthropic")


# =============================================================================
# Exception Classification
# =============================================================================


class TestExceptionClassification:
    def test_timeout_error(self):
        assert _classify_exception(TimeoutError()) == ErrorType.TIMEOUT

    def test_asyncio_timeout(self):
        assert _classify_exception(asyncio.TimeoutError()) == ErrorType.TIMEOUT

    def test_connection_error(self):
        assert _classify_exception(ConnectionError("refused")) == ErrorType.CONNECTION

    def test_rate_limit_in_message(self):
        assert _classify_exception(Exception("rate limit exceeded")) == ErrorType.RATE_LIMIT

    def test_429_in_message(self):
        assert _classify_exception(Exception("HTTP 429")) == ErrorType.RATE_LIMIT

    def test_quota_in_message(self):
        assert _classify_exception(Exception("quota exceeded")) == ErrorType.QUOTA

    def test_billing_in_message(self):
        assert _classify_exception(Exception("billing issue")) == ErrorType.QUOTA

    def test_auth_401(self):
        assert _classify_exception(Exception("HTTP 401 Unauthorized")) == ErrorType.AUTH

    def test_auth_403(self):
        assert _classify_exception(Exception("HTTP 403 Forbidden")) == ErrorType.AUTH

    def test_circuit_open(self):
        assert _classify_exception(Exception("circuit breaker open")) == ErrorType.CIRCUIT_OPEN

    def test_stream_error(self):
        e = type("StreamError", (Exception,), {})()
        assert _classify_exception(e) == ErrorType.STREAM_ERROR

    def test_api_error(self):
        e = type("APIError", (Exception,), {})()
        assert _classify_exception(e) == ErrorType.API_ERROR

    def test_unknown_error(self):
        assert _classify_exception(ValueError("something")) == ErrorType.UNKNOWN


# =============================================================================
# AgentCallTracker
# =============================================================================


class TestAgentCallTracker:
    @pytest.fixture(autouse=True)
    def _init_noop(self):
        with patch("aragora.observability.metrics.agents.get_metrics_enabled", return_value=False):
            init_agent_provider_metrics()

    def test_tracker_defaults(self):
        tracker = AgentCallTracker(provider="anthropic")
        assert tracker.provider == "anthropic"
        assert tracker.model is None
        assert tracker._success is True

    def test_record_tokens(self):
        tracker = AgentCallTracker(provider="openai")
        tracker.record_tokens(input_tokens=100, output_tokens=50)
        assert tracker._input_tokens == 100
        assert tracker._output_tokens == 50

    def test_record_error_enum(self):
        tracker = AgentCallTracker(provider="anthropic")
        tracker.record_error(ErrorType.TIMEOUT)
        assert tracker._success is False
        assert tracker._error_type == ErrorType.TIMEOUT

    def test_record_error_string_valid(self):
        tracker = AgentCallTracker(provider="openai")
        tracker.record_error("rate_limit")
        assert tracker._error_type == ErrorType.RATE_LIMIT

    def test_record_error_string_invalid(self):
        tracker = AgentCallTracker(provider="openai")
        tracker.record_error("some_custom_error")
        assert tracker._error_type == ErrorType.UNKNOWN

    def test_finalize(self):
        tracker = AgentCallTracker(provider="anthropic", model="claude-3")
        tracker.record_tokens(50, 25)
        tracker._finalize()


# =============================================================================
# Context Managers
# =============================================================================


class TestContextManagers:
    @pytest.fixture(autouse=True)
    def _init_noop(self):
        with patch("aragora.observability.metrics.agents.get_metrics_enabled", return_value=False):
            init_agent_provider_metrics()

    def test_sync_context_manager_success(self):
        with track_agent_provider_call("anthropic") as tracker:
            tracker.record_tokens(100, 50)

    def test_sync_context_manager_error(self):
        with pytest.raises(ValueError):
            with track_agent_provider_call("openai") as tracker:
                raise ValueError("test error")

    @pytest.mark.asyncio
    async def test_async_context_manager_success(self):
        async with track_agent_provider_call_async("anthropic") as tracker:
            tracker.record_tokens(100, 50)

    @pytest.mark.asyncio
    async def test_async_context_manager_error(self):
        with pytest.raises(ValueError):
            async with track_agent_provider_call_async("openai") as tracker:
                raise ValueError("test error")


# =============================================================================
# Decorators
# =============================================================================


class TestDecorators:
    @pytest.fixture(autouse=True)
    def _init_noop(self):
        with patch("aragora.observability.metrics.agents.get_metrics_enabled", return_value=False):
            init_agent_provider_metrics()

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        @with_agent_provider_metrics("openai", model="gpt-4")
        async def call_api():
            return "result"

        result = await call_api()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_async_decorator_with_usage(self):
        class MockUsage:
            input_tokens = 100
            output_tokens = 50

        class MockResponse:
            usage = MockUsage()

        @with_agent_provider_metrics("anthropic")
        async def call_api():
            return MockResponse()

        result = await call_api()
        assert result.usage.input_tokens == 100

    @pytest.mark.asyncio
    async def test_async_decorator_openai_style_usage(self):
        class MockUsage:
            prompt_tokens = 200
            completion_tokens = 80

        class MockResponse:
            usage = MockUsage()

        @with_agent_provider_metrics("openai")
        async def call_api():
            return MockResponse()

        result = await call_api()
        assert result.usage.prompt_tokens == 200

    def test_sync_decorator(self):
        @with_agent_provider_metrics_sync("openai")
        def call_api():
            return "sync_result"

        result = call_api()
        assert result == "sync_result"

    def test_sync_decorator_with_usage(self):
        class MockUsage:
            input_tokens = 50
            output_tokens = 20

        class MockResponse:
            usage = MockUsage()

        @with_agent_provider_metrics_sync("anthropic")
        def call_api():
            return MockResponse()

        result = call_api()
        assert result.usage.input_tokens == 50
