"""
Agent Metrics for Aragora.

Tracks agent API requests, latency, and token usage.
"""

from __future__ import annotations

from .types import Counter, Histogram

_STATUS_SUCCESS = "success"
_STATUS_ERROR = "error"
_TOKEN_DIRECTION_INPUT = "input"
_TOKEN_DIRECTION_OUTPUT = "output"
_LATENCY_BUCKETS = (0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

# =============================================================================
# Agent Metrics
# =============================================================================

AGENT_REQUESTS = Counter(
    name="aragora_agent_requests_total",
    help="Agent API requests by agent and status",
    label_names=["agent", "status"],
)

AGENT_LATENCY = Histogram(
    name="aragora_agent_latency_seconds",
    help="Agent response latency",
    label_names=["agent"],
    buckets=list(_LATENCY_BUCKETS),
)

AGENT_TOKENS = Counter(
    name="aragora_agent_tokens_total",
    help="Tokens used by agent",
    label_names=["agent", "direction"],  # direction: input/output
)


# =============================================================================
# Helpers
# =============================================================================


def track_agent_call(
    agent: str,
    latency: float,
    tokens_in: int,
    tokens_out: int,
    success: bool,
    *,
    request_counter: Counter = AGENT_REQUESTS,
    latency_histogram: Histogram = AGENT_LATENCY,
    token_counter: Counter = AGENT_TOKENS,
) -> None:
    """Track an agent API call.

    Optional metric arguments allow focused unit tests to pass fakes without
    mutating module-level collectors.
    """
    status = _STATUS_SUCCESS if success else _STATUS_ERROR

    request_counter.inc(agent=agent, status=status)
    latency_histogram.observe(latency, agent=agent)
    token_counter.inc(tokens_in, agent=agent, direction=_TOKEN_DIRECTION_INPUT)
    token_counter.inc(tokens_out, agent=agent, direction=_TOKEN_DIRECTION_OUTPUT)


__all__ = [
    "AGENT_REQUESTS",
    "AGENT_LATENCY",
    "AGENT_TOKENS",
    "track_agent_call",
]
