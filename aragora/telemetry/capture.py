"""
Per-request telemetry capture pipeline for AI model requests.

Captures tokens, latency, model version, and cost for every AI model call
during debate sessions. Provides:

- ``RequestMetrics`` dataclass for per-request data
- ``DebateSessionMetrics`` for session-level aggregation
- ``capture_model_request`` decorator for automatic instrumentation
- ``RequestMetricsCollector`` for pluggable metric sinks

Usage::

    from aragora.telemetry.capture import capture_model_request, get_session_metrics

    class MyAgent:
        @capture_model_request(provider="anthropic")
        async def generate(self, prompt: str, context=None) -> str:
            ...

    # Retrieve session-level metrics
    metrics = get_session_metrics("debate-123")
"""

from __future__ import annotations

__all__ = [
    "RequestMetrics",
    "DebateSessionMetrics",
    "RequestMetricsCollector",
    "capture_model_request",
    "get_session_metrics",
    "start_session",
    "end_session",
    "register_metrics_sink",
    "unregister_metrics_sink",
    "reset_capture_state",
]

import asyncio
import functools
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, ParamSpec, TypeVar, cast
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class RequestMetrics:
    """Per-request telemetry captured from a single AI model call."""

    request_id: str
    provider: str
    model: str
    agent_name: str
    operation: str  # "generate", "critique", "vote", "revise"

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0

    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Cost (USD)
    cost_usd: float = 0.0

    # Status
    success: bool = True
    error_type: str | None = None

    # Context
    debate_id: str | None = None
    round_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(
        self,
        *,
        success: bool = True,
        error: Exception | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Finalize the metrics after the request completes."""
        self.end_time = time.monotonic()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens

        if error:
            self.error_type = type(error).__name__

        # Calculate cost
        self.cost_usd = _calculate_cost(self.provider, self.model, input_tokens, output_tokens)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DebateSessionMetrics:
    """Aggregated telemetry for a debate session."""

    debate_id: str
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None

    # Aggregated counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    total_cost_usd: float = 0.0

    total_latency_ms: float = 0.0

    # Per-provider breakdown
    provider_stats: dict[str, _ProviderStats] = field(default_factory=dict)

    # Per-model breakdown
    model_stats: dict[str, _ModelStats] = field(default_factory=dict)

    # All captured requests
    requests: list[RequestMetrics] = field(default_factory=list)

    def record(self, metrics: RequestMetrics) -> None:
        """Record a completed request into session aggregates."""
        self.requests.append(metrics)
        self.total_requests += 1
        if metrics.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_input_tokens += metrics.input_tokens
        self.total_output_tokens += metrics.output_tokens
        self.total_tokens += metrics.total_tokens
        self.total_cost_usd += metrics.cost_usd
        self.total_latency_ms += metrics.latency_ms

        # Provider stats
        ps = self.provider_stats.setdefault(
            metrics.provider, _ProviderStats(provider=metrics.provider)
        )
        ps.requests += 1
        ps.input_tokens += metrics.input_tokens
        ps.output_tokens += metrics.output_tokens
        ps.cost_usd += metrics.cost_usd
        ps.total_latency_ms += metrics.latency_ms

        # Model stats
        ms = self.model_stats.setdefault(
            metrics.model, _ModelStats(model=metrics.model, provider=metrics.provider)
        )
        ms.requests += 1
        ms.input_tokens += metrics.input_tokens
        ms.output_tokens += metrics.output_tokens
        ms.cost_usd += metrics.cost_usd

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging or API responses."""
        return {
            "debate_id": self.debate_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "providers": {
                k: {"requests": v.requests, "cost_usd": round(v.cost_usd, 6)}
                for k, v in self.provider_stats.items()
            },
            "models": {
                k: {"requests": v.requests, "cost_usd": round(v.cost_usd, 6)}
                for k, v in self.model_stats.items()
            },
        }


@dataclass
class _ProviderStats:
    provider: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    total_latency_ms: float = 0.0


@dataclass
class _ModelStats:
    model: str
    provider: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


# =============================================================================
# Cost Calculation
# =============================================================================


def _calculate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost using the billing module's pricing table."""
    try:
        from aragora.billing.usage import calculate_token_cost

        cost = calculate_token_cost(provider, model, input_tokens, output_tokens)
        return float(cost)
    except (ImportError, Exception):
        # Fallback: rough estimate at $2/1M input, $8/1M output
        return (input_tokens * 2.0 + output_tokens * 8.0) / 1_000_000


# =============================================================================
# Global State (thread-safe)
# =============================================================================

_lock = threading.Lock()
_sessions: dict[str, DebateSessionMetrics] = {}
_sinks: list[Callable[[RequestMetrics], None]] = []
_request_counter: int = 0


def _next_request_id() -> str:
    global _request_counter
    with _lock:
        _request_counter += 1
        return f"req-{_request_counter}"


# =============================================================================
# Session Management
# =============================================================================


def start_session(debate_id: str) -> DebateSessionMetrics:
    """Start tracking a debate session. Returns the session metrics object."""
    with _lock:
        session = DebateSessionMetrics(debate_id=debate_id)
        _sessions[debate_id] = session
    logger.debug("telemetry_capture_session_started debate_id=%s", debate_id)
    return session


def end_session(debate_id: str) -> DebateSessionMetrics | None:
    """End a session and return its final metrics. Returns None if not found."""
    with _lock:
        session = _sessions.pop(debate_id, None)
    if session:
        session.end_time = time.monotonic()
        logger.debug(
            "telemetry_capture_session_ended debate_id=%s requests=%d cost=%.6f",
            debate_id,
            session.total_requests,
            session.total_cost_usd,
        )
    return session


def get_session_metrics(debate_id: str) -> DebateSessionMetrics | None:
    """Get metrics for an active or completed session."""
    with _lock:
        return _sessions.get(debate_id)


# =============================================================================
# Sink Registration
# =============================================================================


RequestMetricsCollector = Callable[[RequestMetrics], None]


def register_metrics_sink(sink: RequestMetricsCollector) -> None:
    """Register a sink to receive per-request metrics. Thread-safe."""
    with _lock:
        if sink not in _sinks:
            _sinks.append(sink)


def unregister_metrics_sink(sink: RequestMetricsCollector) -> None:
    """Unregister a metrics sink. Thread-safe."""
    with _lock:
        if sink in _sinks:
            _sinks.remove(sink)


def _emit(metrics: RequestMetrics) -> None:
    """Emit metrics to all registered sinks and the active session."""
    # Record into session
    if metrics.debate_id:
        with _lock:
            session = _sessions.get(metrics.debate_id)
        if session:
            session.record(metrics)

    # Emit to sinks
    with _lock:
        sinks = _sinks.copy()
    for sink in sinks:
        try:
            sink(metrics)
        except Exception as e:
            logger.warning("telemetry_capture_sink_error sink=%s error=%s", sink, e)


# =============================================================================
# Decorator
# =============================================================================


def capture_model_request(
    provider: str | None = None,
    operation: str = "generate",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to capture per-request telemetry for AI model calls.

    Automatically extracts agent name, model, and debate context from ``self``.
    Token counts are extracted from the API response when available via
    ``self._last_tokens_in`` / ``self._last_tokens_out`` (set by APIAgent base).

    Args:
        provider: Provider name override. If None, reads from ``self.agent_type``.
        operation: Operation type (generate, critique, vote, revise).

    Usage::

        class MyAgent(APIAgent):
            @capture_model_request(provider="anthropic")
            async def generate(self, prompt, context=None):
                ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            agent_self = args[0] if args else None

            req_provider = provider or getattr(agent_self, "agent_type", "unknown")
            model = getattr(agent_self, "model", "unknown")
            agent_name = getattr(agent_self, "name", "unknown")
            debate_id = getattr(agent_self, "_current_debate_id", None)
            round_number = getattr(agent_self, "_current_round", None)

            # Snapshot token counters before the call
            tokens_in_before = getattr(agent_self, "_total_tokens_in", 0)
            tokens_out_before = getattr(agent_self, "_total_tokens_out", 0)

            metrics = RequestMetrics(
                request_id=_next_request_id(),
                provider=req_provider,
                model=model,
                agent_name=agent_name,
                operation=operation,
                start_time=time.monotonic(),
                debate_id=debate_id,
                round_number=round_number,
            )

            try:
                result = await cast(Awaitable[T], func(*args, **kwargs))

                # Extract token counts from the agent's internal counters
                tokens_in_after = getattr(agent_self, "_total_tokens_in", 0)
                tokens_out_after = getattr(agent_self, "_total_tokens_out", 0)
                input_tokens = tokens_in_after - tokens_in_before
                output_tokens = tokens_out_after - tokens_out_before

                # Fallback: estimate from result text if no counters moved
                if input_tokens == 0 and output_tokens == 0 and isinstance(result, str):
                    # Use first positional arg after self as prompt
                    prompt_text = args[1] if len(args) > 1 else ""
                    input_tokens = len(str(prompt_text)) // 4
                    output_tokens = len(result) // 4

                metrics.complete(
                    success=True,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                return result

            except BaseException as e:
                metrics.complete(
                    success=False,
                    error=e if isinstance(e, Exception) else None,
                )
                raise

            finally:
                _emit(metrics)
                logger.debug(
                    "telemetry_capture agent=%s model=%s op=%s latency=%.0fms "
                    "tokens=%d cost=%.6f success=%s",
                    agent_name,
                    model,
                    operation,
                    metrics.latency_ms,
                    metrics.total_tokens,
                    metrics.cost_usd,
                    metrics.success,
                )

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            agent_self = args[0] if args else None

            req_provider = provider or getattr(agent_self, "agent_type", "unknown")
            model = getattr(agent_self, "model", "unknown")
            agent_name = getattr(agent_self, "name", "unknown")
            debate_id = getattr(agent_self, "_current_debate_id", None)

            tokens_in_before = getattr(agent_self, "_total_tokens_in", 0)
            tokens_out_before = getattr(agent_self, "_total_tokens_out", 0)

            metrics = RequestMetrics(
                request_id=_next_request_id(),
                provider=req_provider,
                model=model,
                agent_name=agent_name,
                operation=operation,
                start_time=time.monotonic(),
                debate_id=debate_id,
            )

            try:
                result = func(*args, **kwargs)

                tokens_in_after = getattr(agent_self, "_total_tokens_in", 0)
                tokens_out_after = getattr(agent_self, "_total_tokens_out", 0)
                input_tokens = tokens_in_after - tokens_in_before
                output_tokens = tokens_out_after - tokens_out_before

                if input_tokens == 0 and output_tokens == 0 and isinstance(result, str):
                    prompt_text = args[1] if len(args) > 1 else ""
                    input_tokens = len(str(prompt_text)) // 4
                    output_tokens = len(result) // 4

                metrics.complete(
                    success=True,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                return result

            except BaseException as e:
                metrics.complete(
                    success=False,
                    error=e if isinstance(e, Exception) else None,
                )
                raise

            finally:
                _emit(metrics)

        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, T], async_wrapper)
        return cast(Callable[P, T], sync_wrapper)

    return decorator


# =============================================================================
# Cleanup (for testing)
# =============================================================================


def reset_capture_state() -> None:
    """Reset all capture state. For testing only."""
    global _sessions, _sinks, _request_counter
    with _lock:
        _sessions = {}
        _sinks = []
        _request_counter = 0
