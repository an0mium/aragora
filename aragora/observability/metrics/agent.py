"""
Agent/LLM call metrics.

Provides Prometheus metrics for tracking agent invocations,
latency, token usage, and error rates.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
AGENT_CALLS: Any = None
AGENT_LATENCY: Any = None
AGENT_ERRORS: Any = None
AGENT_TOKEN_USAGE: Any = None

_initialized = False


def init_agent_metrics() -> None:
    """Initialize agent metrics."""
    global _initialized
    global AGENT_CALLS, AGENT_LATENCY, AGENT_ERRORS, AGENT_TOKEN_USAGE

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Histogram

        AGENT_CALLS = Counter(
            "aragora_agent_calls_total",
            "Total agent/LLM calls",
            ["agent", "status"],
        )

        AGENT_LATENCY = Histogram(
            "aragora_agent_latency_seconds",
            "Agent call latency",
            ["agent"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        AGENT_ERRORS = Counter(
            "aragora_agent_errors_total",
            "Agent errors by type",
            ["agent", "error_type"],
        )

        AGENT_TOKEN_USAGE = Counter(
            "aragora_agent_tokens_total",
            "Token usage by agent",
            ["agent", "direction"],  # direction: input, output
        )

        _initialized = True
        logger.debug("Agent metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global AGENT_CALLS, AGENT_LATENCY, AGENT_ERRORS, AGENT_TOKEN_USAGE

    AGENT_CALLS = NoOpMetric()
    AGENT_LATENCY = NoOpMetric()
    AGENT_ERRORS = NoOpMetric()
    AGENT_TOKEN_USAGE = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_agent_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_agent_call(
    agent: str,
    success: bool,
    latency_seconds: float | None = None,
) -> None:
    """Record an agent/LLM call.

    Args:
        agent: Agent name
        success: Whether the call succeeded
        latency_seconds: Optional call latency
    """
    _ensure_init()
    status = "success" if success else "error"
    AGENT_CALLS.labels(agent=agent, status=status).inc()
    if latency_seconds is not None:
        AGENT_LATENCY.labels(agent=agent).observe(latency_seconds)


def record_agent_latency(agent: str, latency_seconds: float) -> None:
    """Record agent call latency.

    Args:
        agent: Agent name
        latency_seconds: Call latency
    """
    _ensure_init()
    AGENT_LATENCY.labels(agent=agent).observe(latency_seconds)


def record_agent_error(agent: str, error_type: str) -> None:
    """Record an agent error.

    Args:
        agent: Agent name
        error_type: Type of error (timeout, rate_limit, api_error, etc.)
    """
    _ensure_init()
    AGENT_ERRORS.labels(agent=agent, error_type=error_type).inc()


def record_token_usage(
    agent: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record token usage for an agent call.

    Args:
        agent: Agent name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    """
    _ensure_init()
    if input_tokens > 0:
        AGENT_TOKEN_USAGE.labels(agent=agent, direction="input").inc(input_tokens)
    if output_tokens > 0:
        AGENT_TOKEN_USAGE.labels(agent=agent, direction="output").inc(output_tokens)


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def track_agent_call(agent: str) -> Generator[None, None, None]:
    """Context manager to track agent call.

    Automatically records success/failure and latency.

    Args:
        agent: Agent name

    Example:
        with track_agent_call("claude"):
            response = await client.messages.create(...)
    """
    _ensure_init()
    start = time.perf_counter()
    success = True
    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_agent_call(agent, success, latency)


__all__ = [
    # Metrics
    "AGENT_CALLS",
    "AGENT_LATENCY",
    "AGENT_ERRORS",
    "AGENT_TOKEN_USAGE",
    # Functions
    "init_agent_metrics",
    "record_agent_call",
    "record_agent_latency",
    "record_agent_error",
    "record_token_usage",
    "track_agent_call",
]
