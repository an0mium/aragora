"""
Telemetry capture pipeline for AI model requests during debate sessions.

Provides middleware and decorators to capture and log AI model usage metrics
(tokens, latency, model version, cost) tied to specific debate sessions and
decisions.

Usage:
    # As a decorator on agent methods
    @capture_model_request(provider="anthropic")
    async def generate(self, prompt: str) -> str:
        ...

    # As a context manager
    async with model_request_span(
        provider="anthropic",
        model="claude-opus-4-6",
        debate_id="d-123",
        decision_id="dec-456",
    ) as span:
        response = await client.messages.create(...)
        span.record_usage(
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
        )

    # Query session metrics
    store = get_session_metrics_store()
    metrics = store.get_debate_metrics("d-123")
"""

from __future__ import annotations

__all__ = [
    "ModelRequestMetric",
    "SessionMetricsStore",
    "capture_model_request",
    "get_session_metrics_store",
    "model_request_span",
    "reset_session_metrics_store",
]

import asyncio
import functools
import logging
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, ParamSpec, TypeVar, cast
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from uuid import uuid4

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModelRequestMetric:
    """Single AI model request metric tied to a debate session."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Session binding
    debate_id: str | None = None
    session_id: str | None = None
    decision_id: str | None = None
    round_number: int | None = None

    # Provider info
    provider: str = "unknown"
    model: str = "unknown"
    model_version: str | None = None
    agent_name: str = "unknown"

    # Operation
    operation: str = "generate"  # generate, critique, vote, revise, etc.

    # Token usage
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0

    # Timing
    latency_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    # Cost
    cost_usd: Decimal = field(default_factory=lambda: Decimal("0"))

    # Status
    success: bool = True
    error_type: str | None = None
    error_message: str | None = None

    # Extensible metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_cost(self) -> Decimal:
        """Calculate cost using billing module and store it."""
        try:
            from aragora.billing.usage import calculate_token_cost

            self.cost_usd = calculate_token_cost(
                provider=self.provider,
                model=self.model,
                tokens_in=self.tokens_in,
                tokens_out=self.tokens_out,
                tokens_cached=self.tokens_cached,
            )
        except (ImportError, Exception) as exc:
            logger.debug("cost_calculation_failed: %s", exc)
            self.cost_usd = Decimal("0")
        return self.cost_usd

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_usd"] = float(self.cost_usd)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# ---------------------------------------------------------------------------
# Session-scoped metrics store
# ---------------------------------------------------------------------------


class SessionMetricsStore:
    """Thread-safe store for model request metrics scoped to debate sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: list[ModelRequestMetric] = []

    def record(self, metric: ModelRequestMetric) -> None:
        with self._lock:
            self._metrics.append(metric)

    def get_all(self) -> list[ModelRequestMetric]:
        with self._lock:
            return list(self._metrics)

    def get_debate_metrics(self, debate_id: str) -> list[ModelRequestMetric]:
        with self._lock:
            return [m for m in self._metrics if m.debate_id == debate_id]

    def get_decision_metrics(self, decision_id: str) -> list[ModelRequestMetric]:
        with self._lock:
            return [m for m in self._metrics if m.decision_id == decision_id]

    def get_session_metrics(self, session_id: str) -> list[ModelRequestMetric]:
        with self._lock:
            return [m for m in self._metrics if m.session_id == session_id]

    def summarize_debate(self, debate_id: str) -> dict[str, Any]:
        """Aggregate metrics summary for a debate."""
        metrics = self.get_debate_metrics(debate_id)
        if not metrics:
            return {
                "debate_id": debate_id,
                "total_requests": 0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
                "total_cost_usd": 0.0,
                "total_latency_ms": 0.0,
                "providers": {},
                "models": {},
                "agents": {},
            }

        providers: dict[str, int] = {}
        models: dict[str, int] = {}
        agents: dict[str, int] = {}
        total_tokens_in = 0
        total_tokens_out = 0
        total_cost = Decimal("0")
        total_latency = 0.0
        success_count = 0

        for m in metrics:
            total_tokens_in += m.tokens_in
            total_tokens_out += m.tokens_out
            total_cost += m.cost_usd
            total_latency += m.latency_ms
            if m.success:
                success_count += 1
            providers[m.provider] = providers.get(m.provider, 0) + 1
            models[m.model] = models.get(m.model, 0) + 1
            agents[m.agent_name] = agents.get(m.agent_name, 0) + 1

        return {
            "debate_id": debate_id,
            "total_requests": len(metrics),
            "successful_requests": success_count,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_cost_usd": float(total_cost),
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency / len(metrics) if metrics else 0.0,
            "providers": providers,
            "models": models,
            "agents": agents,
        }

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()


# Singleton store
_store_lock = threading.Lock()
_store: SessionMetricsStore | None = None


def get_session_metrics_store() -> SessionMetricsStore:
    """Get the global session metrics store (singleton)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = SessionMetricsStore()
        return _store


def reset_session_metrics_store() -> None:
    """Reset the global store (for testing)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.clear()
        _store = None


# ---------------------------------------------------------------------------
# Emission helpers – push metrics to Prometheus / observability
# ---------------------------------------------------------------------------


def _emit_to_observability(metric: ModelRequestMetric) -> None:
    """Push a completed metric to Prometheus and tracing subsystems."""
    try:
        from aragora.observability.metrics.agents import (
            record_provider_call,
            record_provider_token_usage,
        )

        record_provider_call(
            provider=metric.provider,
            success=metric.success,
            latency_seconds=metric.latency_ms / 1000.0 if metric.latency_ms else None,
            model=metric.model,
        )
        if metric.tokens_in > 0 or metric.tokens_out > 0:
            record_provider_token_usage(
                provider=metric.provider,
                input_tokens=metric.tokens_in,
                output_tokens=metric.tokens_out,
            )
    except ImportError:
        pass


def _finalize_metric(metric: ModelRequestMetric) -> None:
    """Calculate cost, store, and emit a completed metric."""
    metric.calculate_cost()
    get_session_metrics_store().record(metric)
    _emit_to_observability(metric)
    logger.debug(
        "model_request provider=%s model=%s debate=%s tokens_in=%d tokens_out=%d "
        "latency=%.0fms cost=$%.6f success=%s",
        metric.provider,
        metric.model,
        metric.debate_id,
        metric.tokens_in,
        metric.tokens_out,
        metric.latency_ms,
        float(metric.cost_usd),
        metric.success,
    )


# ---------------------------------------------------------------------------
# Context-manager span
# ---------------------------------------------------------------------------


class _ModelRequestSpan:
    """Mutable handle yielded by the context manager."""

    def __init__(self, metric: ModelRequestMetric) -> None:
        self._metric = metric

    def record_usage(
        self,
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_cached: int = 0,
    ) -> None:
        self._metric.tokens_in = tokens_in
        self._metric.tokens_out = tokens_out
        self._metric.tokens_cached = tokens_cached

    def set_model(self, model: str, version: str | None = None) -> None:
        self._metric.model = model
        if version:
            self._metric.model_version = version

    def set_metadata(self, key: str, value: Any) -> None:
        self._metric.metadata[key] = value

    @property
    def metric(self) -> ModelRequestMetric:
        return self._metric


@contextmanager
def model_request_span(
    provider: str,
    model: str = "unknown",
    *,
    debate_id: str | None = None,
    session_id: str | None = None,
    decision_id: str | None = None,
    round_number: int | None = None,
    agent_name: str = "unknown",
    operation: str = "generate",
) -> Generator[_ModelRequestSpan, None, None]:
    """Sync context manager that captures a single model request metric."""
    metric = ModelRequestMetric(
        provider=provider,
        model=model,
        debate_id=debate_id,
        session_id=session_id,
        decision_id=decision_id,
        round_number=round_number,
        agent_name=agent_name,
        operation=operation,
        start_time=time.perf_counter(),
    )
    span = _ModelRequestSpan(metric)
    try:
        yield span
    except Exception as exc:
        metric.success = False
        metric.error_type = type(exc).__name__
        metric.error_message = str(exc)[:200]
        raise
    finally:
        metric.end_time = time.perf_counter()
        metric.latency_ms = (metric.end_time - metric.start_time) * 1000
        _finalize_metric(metric)


@asynccontextmanager
async def async_model_request_span(
    provider: str,
    model: str = "unknown",
    *,
    debate_id: str | None = None,
    session_id: str | None = None,
    decision_id: str | None = None,
    round_number: int | None = None,
    agent_name: str = "unknown",
    operation: str = "generate",
) -> AsyncGenerator[_ModelRequestSpan, None]:
    """Async context manager that captures a single model request metric."""
    metric = ModelRequestMetric(
        provider=provider,
        model=model,
        debate_id=debate_id,
        session_id=session_id,
        decision_id=decision_id,
        round_number=round_number,
        agent_name=agent_name,
        operation=operation,
        start_time=time.perf_counter(),
    )
    span = _ModelRequestSpan(metric)
    try:
        yield span
    except Exception as exc:
        metric.success = False
        metric.error_type = type(exc).__name__
        metric.error_message = str(exc)[:200]
        raise
    finally:
        metric.end_time = time.perf_counter()
        metric.latency_ms = (metric.end_time - metric.start_time) * 1000
        _finalize_metric(metric)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def capture_model_request(
    provider: str | None = None,
    operation: str = "generate",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that captures model request telemetry for debate-bound calls.

    Attempts to extract debate context and model info from ``self`` attributes:
    - ``self.provider`` or falls back to *provider* arg
    - ``self.model``
    - ``self.name`` for agent_name
    - ``self.debate_id``, ``self.session_id``, ``self.decision_id``

    After the call, if the result exposes a ``usage`` attribute (common for
    Anthropic/OpenAI SDK responses), tokens are extracted automatically.

    Args:
        provider: Default provider name. Overridden by ``self.provider``.
        operation: Operation label (generate, critique, vote, revise).
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        def _extract_context(args: tuple, kwargs: dict) -> dict[str, Any]:
            ctx: dict[str, Any] = {"provider": provider or "unknown", "operation": operation}
            if args and hasattr(args[0], "__dict__"):
                obj = args[0]
                ctx["provider"] = getattr(obj, "provider", ctx["provider"])
                ctx["model"] = getattr(obj, "model", "unknown")
                ctx["agent_name"] = getattr(obj, "name", "unknown")
                ctx["debate_id"] = getattr(obj, "debate_id", None)
                ctx["session_id"] = getattr(obj, "session_id", None)
                ctx["decision_id"] = getattr(obj, "decision_id", None)
                ctx["round_number"] = getattr(obj, "round_number", None)
            return ctx

        def _extract_usage(result: Any, metric: ModelRequestMetric) -> None:
            if hasattr(result, "usage"):
                usage = result.usage
                metric.tokens_in = getattr(usage, "input_tokens", 0) or getattr(
                    usage, "prompt_tokens", 0
                )
                metric.tokens_out = getattr(usage, "output_tokens", 0) or getattr(
                    usage, "completion_tokens", 0
                )
                metric.tokens_cached = getattr(usage, "cached_tokens", 0)
            elif isinstance(result, str):
                # Estimate from text length as fallback
                metric.tokens_out = len(result) // 4

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            ctx = _extract_context(args, kwargs)
            metric = ModelRequestMetric(
                provider=ctx.get("provider", "unknown"),
                model=ctx.get("model", "unknown"),
                agent_name=ctx.get("agent_name", "unknown"),
                operation=ctx.get("operation", operation),
                debate_id=ctx.get("debate_id"),
                session_id=ctx.get("session_id"),
                decision_id=ctx.get("decision_id"),
                round_number=ctx.get("round_number"),
                start_time=time.perf_counter(),
            )
            try:
                result = await cast(Awaitable[T], func(*args, **kwargs))
                _extract_usage(result, metric)
                return result
            except Exception as exc:
                metric.success = False
                metric.error_type = type(exc).__name__
                metric.error_message = str(exc)[:200]
                raise
            finally:
                metric.end_time = time.perf_counter()
                metric.latency_ms = (metric.end_time - metric.start_time) * 1000
                _finalize_metric(metric)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            ctx = _extract_context(args, kwargs)
            metric = ModelRequestMetric(
                provider=ctx.get("provider", "unknown"),
                model=ctx.get("model", "unknown"),
                agent_name=ctx.get("agent_name", "unknown"),
                operation=ctx.get("operation", operation),
                debate_id=ctx.get("debate_id"),
                session_id=ctx.get("session_id"),
                decision_id=ctx.get("decision_id"),
                round_number=ctx.get("round_number"),
                start_time=time.perf_counter(),
            )
            try:
                result = func(*args, **kwargs)
                _extract_usage(result, metric)
                return result
            except Exception as exc:
                metric.success = False
                metric.error_type = type(exc).__name__
                metric.error_message = str(exc)[:200]
                raise
            finally:
                metric.end_time = time.perf_counter()
                metric.latency_ms = (metric.end_time - metric.start_time) * 1000
                _finalize_metric(metric)

        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, T], async_wrapper)
        return cast(Callable[P, T], sync_wrapper)

    return decorator
