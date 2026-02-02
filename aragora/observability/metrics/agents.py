"""
Per-provider agent metrics for AI API providers.

Provides comprehensive Prometheus metrics for tracking AI provider performance,
including calls, latency, token usage, connection pooling, rate limits,
fallback chains, and circuit breaker states.

Usage:
    from aragora.observability.metrics.agents import (
        track_agent_provider_call,
        record_provider_token_usage,
        set_circuit_breaker_state,
    )

    # As context manager
    async with track_agent_provider_call("anthropic") as tracker:
        response = await client.messages.create(...)
        tracker.record_tokens(input_tokens=100, output_tokens=50)

    # As decorator
    @with_agent_provider_metrics("openai")
    async def call_openai(prompt: str) -> str:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    ParamSpec,
    TypeVar,
)

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# Enums and Types
# =============================================================================


class CircuitBreakerState(str, Enum):
    """Circuit breaker states for metrics reporting."""

    OPEN = "open"
    CLOSED = "closed"
    HALF_OPEN = "half_open"


class ErrorType(str, Enum):
    """Standardized error types for metrics."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    AUTH = "auth"
    CONNECTION = "connection"
    API_ERROR = "api_error"
    STREAM_ERROR = "stream_error"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


class TokenType(str, Enum):
    """Token types for usage tracking."""

    INPUT = "input"
    OUTPUT = "output"


# =============================================================================
# Global Metrics Variables
# =============================================================================

# Call metrics
AGENT_PROVIDER_CALLS: Any = None
AGENT_PROVIDER_LATENCY: Any = None

# Token metrics
AGENT_TOKEN_USAGE: Any = None

# Connection pool metrics
AGENT_CONNECTION_POOL_ACTIVE: Any = None
AGENT_CONNECTION_POOL_WAITING: Any = None

# Rate limit metrics
AGENT_RATE_LIMIT_DETECTED: Any = None
AGENT_RATE_LIMIT_BACKOFF_SECONDS: Any = None

# Fallback metrics
AGENT_FALLBACK_CHAIN_DEPTH: Any = None
AGENT_FALLBACK_TRIGGERED: Any = None

# Circuit breaker metrics
AGENT_CIRCUIT_BREAKER_STATE: Any = None
AGENT_CIRCUIT_BREAKER_STATE_CHANGES: Any = None
AGENT_CIRCUIT_BREAKER_REJECTIONS: Any = None

# Model-specific metrics
AGENT_MODEL_CALLS: Any = None

_initialized = False


# =============================================================================
# Initialization
# =============================================================================


def init_agent_provider_metrics() -> None:
    """Initialize agent provider metrics."""
    global _initialized
    global AGENT_PROVIDER_CALLS, AGENT_PROVIDER_LATENCY
    global AGENT_TOKEN_USAGE
    global AGENT_CONNECTION_POOL_ACTIVE, AGENT_CONNECTION_POOL_WAITING
    global AGENT_RATE_LIMIT_DETECTED, AGENT_RATE_LIMIT_BACKOFF_SECONDS
    global AGENT_FALLBACK_CHAIN_DEPTH, AGENT_FALLBACK_TRIGGERED
    global AGENT_CIRCUIT_BREAKER_STATE, AGENT_CIRCUIT_BREAKER_STATE_CHANGES
    global AGENT_CIRCUIT_BREAKER_REJECTIONS
    global AGENT_MODEL_CALLS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import REGISTRY, Counter, Gauge, Histogram

        # Helper to get existing metric or create new one
        def get_or_create_counter(name: str, doc: str, labels: list[str]) -> Counter:
            """Get existing counter or create a new one."""
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    # prometheus_client registry stores Collector base type;
                    # the isinstance check is implicit via _name matching.
                    return cast(Counter, collector)
            return Counter(name, doc, labels)

        def get_or_create_histogram(
            name: str, doc: str, labels: list[str], buckets: list[float]
        ) -> Histogram:
            """Get existing histogram or create a new one."""
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    # prometheus_client registry stores Collector base type;
                    # the isinstance check is implicit via _name matching.
                    return cast(Histogram, collector)
            return Histogram(name, doc, labels, buckets=buckets)

        def get_or_create_gauge(name: str, doc: str, labels: list[str]) -> Gauge:
            """Get existing gauge or create a new one."""
            for collector in REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    # prometheus_client registry stores Collector base type;
                    # the isinstance check is implicit via _name matching.
                    return cast(Gauge, collector)
            return Gauge(name, doc, labels)

        # --- Call Metrics ---
        AGENT_PROVIDER_CALLS = get_or_create_counter(
            "aragora_agent_provider_calls_total",
            "Total agent/LLM API calls by provider",
            ["provider", "status", "error_type"],
        )

        AGENT_PROVIDER_LATENCY = get_or_create_histogram(
            "aragora_agent_provider_latency_seconds",
            "Agent call latency by provider",
            ["provider"],
            [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
        )

        # --- Token Metrics ---
        AGENT_TOKEN_USAGE = get_or_create_counter(
            "aragora_agent_token_usage_total",
            "Token usage by provider and type (input/output)",
            ["provider", "token_type"],
        )

        # --- Connection Pool Metrics ---
        AGENT_CONNECTION_POOL_ACTIVE = get_or_create_gauge(
            "aragora_agent_connection_pool_active",
            "Active connections in pool by provider",
            ["provider"],
        )

        AGENT_CONNECTION_POOL_WAITING = get_or_create_gauge(
            "aragora_agent_connection_pool_waiting",
            "Waiting requests for connections by provider",
            ["provider"],
        )

        # --- Rate Limit Metrics ---
        AGENT_RATE_LIMIT_DETECTED = get_or_create_counter(
            "aragora_agent_rate_limit_detected_total",
            "Rate limit events detected by provider",
            ["provider"],
        )

        AGENT_RATE_LIMIT_BACKOFF_SECONDS = get_or_create_histogram(
            "aragora_agent_rate_limit_backoff_seconds",
            "Backoff time applied after rate limits",
            ["provider"],
            [1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        # --- Fallback Metrics ---
        AGENT_FALLBACK_CHAIN_DEPTH = get_or_create_histogram(
            "aragora_agent_fallback_chain_depth",
            "Depth of fallback chain before success",
            [],
            [0, 1, 2, 3, 4, 5],
        )

        AGENT_FALLBACK_TRIGGERED = get_or_create_counter(
            "aragora_agent_fallback_triggered_total",
            "Fallback activations by primary and fallback provider",
            ["primary_provider", "fallback_provider", "trigger_reason"],
        )

        # --- Circuit Breaker Metrics ---
        # Note: Use 'provider_circuit_breaker' prefix to avoid collision with
        # aragora/server/prometheus.py which uses 'agent_circuit_breaker' with 'agent_type' label
        AGENT_CIRCUIT_BREAKER_STATE = get_or_create_gauge(
            "aragora_provider_circuit_breaker_state",
            "Current circuit breaker state (0=closed, 1=half_open, 2=open)",
            ["provider"],
        )

        AGENT_CIRCUIT_BREAKER_STATE_CHANGES = get_or_create_counter(
            "aragora_provider_circuit_breaker_state_changes_total",
            "Circuit breaker state transitions",
            ["provider", "from_state", "to_state"],
        )

        AGENT_CIRCUIT_BREAKER_REJECTIONS = get_or_create_counter(
            "aragora_provider_circuit_breaker_rejections_total",
            "Requests rejected due to open circuit breaker",
            ["provider"],
        )

        # --- Model-Specific Metrics ---
        AGENT_MODEL_CALLS = get_or_create_counter(
            "aragora_agent_model_calls_total",
            "Calls by specific model",
            ["provider", "model", "status"],
        )

        _initialized = True
        logger.debug("Agent provider metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global AGENT_PROVIDER_CALLS, AGENT_PROVIDER_LATENCY
    global AGENT_TOKEN_USAGE
    global AGENT_CONNECTION_POOL_ACTIVE, AGENT_CONNECTION_POOL_WAITING
    global AGENT_RATE_LIMIT_DETECTED, AGENT_RATE_LIMIT_BACKOFF_SECONDS
    global AGENT_FALLBACK_CHAIN_DEPTH, AGENT_FALLBACK_TRIGGERED
    global AGENT_CIRCUIT_BREAKER_STATE, AGENT_CIRCUIT_BREAKER_STATE_CHANGES
    global AGENT_CIRCUIT_BREAKER_REJECTIONS
    global AGENT_MODEL_CALLS

    AGENT_PROVIDER_CALLS = NoOpMetric()
    AGENT_PROVIDER_LATENCY = NoOpMetric()
    AGENT_TOKEN_USAGE = NoOpMetric()
    AGENT_CONNECTION_POOL_ACTIVE = NoOpMetric()
    AGENT_CONNECTION_POOL_WAITING = NoOpMetric()
    AGENT_RATE_LIMIT_DETECTED = NoOpMetric()
    AGENT_RATE_LIMIT_BACKOFF_SECONDS = NoOpMetric()
    AGENT_FALLBACK_CHAIN_DEPTH = NoOpMetric()
    AGENT_FALLBACK_TRIGGERED = NoOpMetric()
    AGENT_CIRCUIT_BREAKER_STATE = NoOpMetric()
    AGENT_CIRCUIT_BREAKER_STATE_CHANGES = NoOpMetric()
    AGENT_CIRCUIT_BREAKER_REJECTIONS = NoOpMetric()
    AGENT_MODEL_CALLS = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_agent_provider_metrics()


# =============================================================================
# Metric Recording Functions
# =============================================================================


def record_provider_call(
    provider: str,
    success: bool,
    error_type: str | ErrorType | None = None,
    latency_seconds: float | None = None,
    model: str | None = None,
) -> None:
    """Record an agent provider API call.

    Args:
        provider: Provider name (anthropic, openai, mistral, grok, openrouter)
        success: Whether the call succeeded
        error_type: Type of error if failed (ErrorType enum or string)
        latency_seconds: Optional call latency
        model: Optional model name for model-specific tracking
    """
    _ensure_init()

    status = "success" if success else "error"
    error_type_str = "none"
    if not success and error_type:
        error_type_str = error_type.value if isinstance(error_type, ErrorType) else str(error_type)

    AGENT_PROVIDER_CALLS.labels(
        provider=provider,
        status=status,
        error_type=error_type_str,
    ).inc()

    if latency_seconds is not None:
        AGENT_PROVIDER_LATENCY.labels(provider=provider).observe(latency_seconds)

    # Model-specific tracking
    if model:
        AGENT_MODEL_CALLS.labels(
            provider=provider,
            model=model,
            status=status,
        ).inc()


def record_provider_latency(provider: str, latency_seconds: float) -> None:
    """Record provider call latency.

    Args:
        provider: Provider name
        latency_seconds: Call latency in seconds
    """
    _ensure_init()
    AGENT_PROVIDER_LATENCY.labels(provider=provider).observe(latency_seconds)


def record_provider_token_usage(
    provider: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record token usage for a provider call.

    Args:
        provider: Provider name
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
    """
    _ensure_init()
    if input_tokens > 0:
        AGENT_TOKEN_USAGE.labels(provider=provider, token_type=TokenType.INPUT.value).inc(
            input_tokens
        )
    if output_tokens > 0:
        AGENT_TOKEN_USAGE.labels(provider=provider, token_type=TokenType.OUTPUT.value).inc(
            output_tokens
        )


def set_connection_pool_active(provider: str, count: int) -> None:
    """Set active connection count for a provider.

    Args:
        provider: Provider name
        count: Number of active connections
    """
    _ensure_init()
    AGENT_CONNECTION_POOL_ACTIVE.labels(provider=provider).set(count)


def set_connection_pool_waiting(provider: str, count: int) -> None:
    """Set waiting request count for a provider.

    Args:
        provider: Provider name
        count: Number of waiting requests
    """
    _ensure_init()
    AGENT_CONNECTION_POOL_WAITING.labels(provider=provider).set(count)


def record_rate_limit_detected(provider: str, backoff_seconds: float | None = None) -> None:
    """Record a rate limit event.

    Args:
        provider: Provider name
        backoff_seconds: Optional backoff time applied
    """
    _ensure_init()
    AGENT_RATE_LIMIT_DETECTED.labels(provider=provider).inc()
    if backoff_seconds is not None:
        AGENT_RATE_LIMIT_BACKOFF_SECONDS.labels(provider=provider).observe(backoff_seconds)


def record_fallback_triggered(
    primary_provider: str,
    fallback_provider: str,
    trigger_reason: str,
) -> None:
    """Record a fallback activation.

    Args:
        primary_provider: Provider that failed
        fallback_provider: Provider used as fallback
        trigger_reason: Reason for fallback (rate_limit, quota, error, etc.)
    """
    _ensure_init()
    AGENT_FALLBACK_TRIGGERED.labels(
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        trigger_reason=trigger_reason,
    ).inc()


def record_fallback_chain_depth(depth: int) -> None:
    """Record depth of fallback chain traversed.

    Args:
        depth: Number of fallback attempts before success (0 = no fallback needed)
    """
    _ensure_init()
    AGENT_FALLBACK_CHAIN_DEPTH.observe(depth)


def set_circuit_breaker_state(
    provider: str,
    state: CircuitBreakerState | str,
) -> None:
    """Set circuit breaker state for a provider.

    Args:
        provider: Provider name
        state: CircuitBreakerState or string (open, closed, half_open)
    """
    _ensure_init()
    if isinstance(state, str):
        state = CircuitBreakerState(state.lower())

    # Numeric mapping: closed=0, half_open=1, open=2
    state_value = {
        CircuitBreakerState.CLOSED: 0,
        CircuitBreakerState.HALF_OPEN: 1,
        CircuitBreakerState.OPEN: 2,
    }[state]

    AGENT_CIRCUIT_BREAKER_STATE.labels(provider=provider).set(state_value)


def record_circuit_breaker_state_change(
    provider: str,
    from_state: CircuitBreakerState | str,
    to_state: CircuitBreakerState | str,
) -> None:
    """Record a circuit breaker state transition.

    Args:
        provider: Provider name
        from_state: Previous state
        to_state: New state
    """
    _ensure_init()
    from_str = from_state.value if isinstance(from_state, CircuitBreakerState) else from_state
    to_str = to_state.value if isinstance(to_state, CircuitBreakerState) else to_state

    AGENT_CIRCUIT_BREAKER_STATE_CHANGES.labels(
        provider=provider,
        from_state=from_str,
        to_state=to_str,
    ).inc()

    # Also update the current state gauge
    set_circuit_breaker_state(provider, to_state)


def record_circuit_breaker_rejection(provider: str) -> None:
    """Record a request rejection due to open circuit breaker.

    Args:
        provider: Provider name
    """
    _ensure_init()
    AGENT_CIRCUIT_BREAKER_REJECTIONS.labels(provider=provider).inc()


# =============================================================================
# Call Tracker Context Manager
# =============================================================================


@dataclass
class AgentCallTracker:
    """Tracks metrics for a single agent API call.

    Used as context manager to automatically record timing and status.
    """

    provider: str
    model: str | None = None
    _start_time: float = field(default_factory=time.perf_counter, init=False)
    _success: bool = field(default=True, init=False)
    _error_type: ErrorType | None = field(default=None, init=False)
    _input_tokens: int = field(default=0, init=False)
    _output_tokens: int = field(default=0, init=False)

    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record token usage for this call."""
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def record_error(self, error_type: ErrorType | str) -> None:
        """Record that this call failed with the given error type."""
        self._success = False
        if isinstance(error_type, str):
            try:
                self._error_type = ErrorType(error_type)
            except ValueError:
                self._error_type = ErrorType.UNKNOWN
        else:
            self._error_type = error_type

    def _finalize(self) -> None:
        """Record all metrics when context exits."""
        latency = time.perf_counter() - self._start_time

        record_provider_call(
            provider=self.provider,
            success=self._success,
            error_type=self._error_type,
            latency_seconds=latency,
            model=self.model,
        )

        if self._input_tokens > 0 or self._output_tokens > 0:
            record_provider_token_usage(
                provider=self.provider,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
            )


@contextmanager
def track_agent_provider_call(
    provider: str,
    model: str | None = None,
) -> Generator[AgentCallTracker, None, None]:
    """Context manager to track agent provider call metrics.

    Automatically records latency, success/failure, and token usage.

    Args:
        provider: Provider name (anthropic, openai, etc.)
        model: Optional model name for detailed tracking

    Yields:
        AgentCallTracker instance for recording tokens and errors

    Example:
        with track_agent_provider_call("anthropic", model="claude-3-opus") as tracker:
            response = await client.messages.create(...)
            tracker.record_tokens(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
    """
    _ensure_init()
    tracker = AgentCallTracker(provider=provider, model=model)
    try:
        yield tracker
    except Exception as e:
        # Classify the exception
        error_type = _classify_exception(e)
        tracker.record_error(error_type)
        raise
    finally:
        tracker._finalize()


@asynccontextmanager
async def track_agent_provider_call_async(
    provider: str,
    model: str | None = None,
) -> AsyncGenerator[AgentCallTracker, None]:
    """Async context manager to track agent provider call metrics.

    Same as track_agent_provider_call but for async contexts.
    """
    _ensure_init()
    tracker = AgentCallTracker(provider=provider, model=model)
    try:
        yield tracker
    except Exception as e:
        error_type = _classify_exception(e)
        tracker.record_error(error_type)
        raise
    finally:
        tracker._finalize()


def _classify_exception(e: Exception) -> ErrorType:
    """Classify an exception into an ErrorType."""
    error_name = type(e).__name__.lower()
    error_str = str(e).lower()

    # Check for rate limit errors
    if "rate" in error_name or "rate" in error_str or "429" in error_str:
        return ErrorType.RATE_LIMIT

    # Check for quota errors
    if "quota" in error_str or "credit" in error_str or "billing" in error_str:
        return ErrorType.QUOTA

    # Check for timeout errors
    if "timeout" in error_name or isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return ErrorType.TIMEOUT

    # Check for connection errors
    if "connection" in error_name or "connect" in error_str:
        return ErrorType.CONNECTION

    # Check for auth errors
    if "auth" in error_name or "401" in error_str or "403" in error_str:
        return ErrorType.AUTH

    # Check for circuit breaker
    if "circuit" in error_name or "circuit" in error_str:
        return ErrorType.CIRCUIT_OPEN

    # Check for stream errors
    if "stream" in error_name:
        return ErrorType.STREAM_ERROR

    # Check for general API errors
    if "api" in error_name:
        return ErrorType.API_ERROR

    return ErrorType.UNKNOWN


# =============================================================================
# Decorator
# =============================================================================


def with_agent_provider_metrics(
    provider: str,
    model: str | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator to add provider metrics to an async function.

    Args:
        provider: Provider name
        model: Optional model name

    Example:
        @with_agent_provider_metrics("openai", model="gpt-4")
        async def call_openai(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with track_agent_provider_call_async(provider, model) as tracker:
                result = await func(*args, **kwargs)

                # Try to extract token usage from result if it has usage info
                if hasattr(result, "usage"):
                    usage = result.usage
                    if hasattr(usage, "input_tokens"):
                        tracker.record_tokens(
                            input_tokens=getattr(usage, "input_tokens", 0),
                            output_tokens=getattr(usage, "output_tokens", 0),
                        )
                    elif hasattr(usage, "prompt_tokens"):
                        tracker.record_tokens(
                            input_tokens=getattr(usage, "prompt_tokens", 0),
                            output_tokens=getattr(usage, "completion_tokens", 0),
                        )

                return result

        return wrapper

    return decorator


def with_agent_provider_metrics_sync(
    provider: str,
    model: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add provider metrics to a sync function.

    Args:
        provider: Provider name
        model: Optional model name

    Example:
        @with_agent_provider_metrics_sync("openai", model="gpt-4")
        def call_openai(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with track_agent_provider_call(provider, model) as tracker:
                result = func(*args, **kwargs)

                # Try to extract token usage from result if it has usage info
                if hasattr(result, "usage"):
                    usage = result.usage
                    if hasattr(usage, "input_tokens"):
                        tracker.record_tokens(
                            input_tokens=getattr(usage, "input_tokens", 0),
                            output_tokens=getattr(usage, "output_tokens", 0),
                        )
                    elif hasattr(usage, "prompt_tokens"):
                        tracker.record_tokens(
                            input_tokens=getattr(usage, "prompt_tokens", 0),
                            output_tokens=getattr(usage, "completion_tokens", 0),
                        )

                return result

        return wrapper

    return decorator


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "CircuitBreakerState",
    "ErrorType",
    "TokenType",
    # Metrics objects (for direct access if needed)
    "AGENT_PROVIDER_CALLS",
    "AGENT_PROVIDER_LATENCY",
    "AGENT_TOKEN_USAGE",
    "AGENT_CONNECTION_POOL_ACTIVE",
    "AGENT_CONNECTION_POOL_WAITING",
    "AGENT_RATE_LIMIT_DETECTED",
    "AGENT_RATE_LIMIT_BACKOFF_SECONDS",
    "AGENT_FALLBACK_CHAIN_DEPTH",
    "AGENT_FALLBACK_TRIGGERED",
    "AGENT_CIRCUIT_BREAKER_STATE",
    "AGENT_CIRCUIT_BREAKER_STATE_CHANGES",
    "AGENT_CIRCUIT_BREAKER_REJECTIONS",
    "AGENT_MODEL_CALLS",
    # Init
    "init_agent_provider_metrics",
    # Recording functions
    "record_provider_call",
    "record_provider_latency",
    "record_provider_token_usage",
    "set_connection_pool_active",
    "set_connection_pool_waiting",
    "record_rate_limit_detected",
    "record_fallback_triggered",
    "record_fallback_chain_depth",
    "set_circuit_breaker_state",
    "record_circuit_breaker_state_change",
    "record_circuit_breaker_rejection",
    # Context managers
    "track_agent_provider_call",
    "track_agent_provider_call_async",
    "AgentCallTracker",
    # Decorators
    "with_agent_provider_metrics",
    "with_agent_provider_metrics_sync",
]
