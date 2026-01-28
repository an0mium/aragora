"""
Agent Metrics for Aragora.

Tracks agent API requests, latency, and token usage.
"""

from __future__ import annotations

from .types import Counter, Histogram

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
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
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
    agent: str, latency: float, tokens_in: int, tokens_out: int, success: bool
) -> None:
    """Track an agent API call."""
    AGENT_REQUESTS.inc(agent=agent, status="success" if success else "error")
    AGENT_LATENCY.observe(latency, agent=agent)
    AGENT_TOKENS.inc(tokens_in, agent=agent, direction="input")
    AGENT_TOKENS.inc(tokens_out, agent=agent, direction="output")


__all__ = [
    "AGENT_REQUESTS",
    "AGENT_LATENCY",
    "AGENT_TOKENS",
    "track_agent_call",
]
