"""
Gateway Metrics for Aragora.

Tracks external agent gateway operations, credential caching, and hybrid verification timing.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from collections.abc import Generator

from .types import Counter, Histogram

logger = logging.getLogger(__name__)

# =============================================================================
# Gateway Metrics
# =============================================================================

GATEWAY_OPERATION_LATENCY = Histogram(
    name="aragora_gateway_operation_latency_seconds",
    help="Latency of gateway operations",
    label_names=["operation", "agent_type", "status"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

GATEWAY_EXTERNAL_CALLS = Counter(
    name="aragora_gateway_external_calls_total",
    help="Total external agent calls",
    label_names=["agent_type", "operation", "status"],
)

CREDENTIAL_CACHE_HITS = Counter(
    name="aragora_gateway_credential_cache_hits_total",
    help="Credential cache hits",
)

CREDENTIAL_CACHE_MISSES = Counter(
    name="aragora_gateway_credential_cache_misses_total",
    help="Credential cache misses",
)

HYBRID_VERIFICATION_TIME = Histogram(
    name="aragora_gateway_hybrid_verification_seconds",
    help="Time for hybrid debate verification",
    label_names=["phase"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)


# =============================================================================
# Helpers
# =============================================================================


@contextmanager
def track_gateway_operation(
    operation: str, agent_type: str = "unknown"
) -> Generator[None, None, None]:
    """Context manager to track gateway operation latency.

    Args:
        operation: Operation type (e.g. generate, critique, refine).
        agent_type: External agent type (e.g. openclaw, crewai).

    Usage:
        with track_gateway_operation("generate", "openclaw"):
            result = await external_agent.generate(prompt)
    """
    start = time.monotonic()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.monotonic() - start
        GATEWAY_OPERATION_LATENCY.observe(
            duration,
            operation=operation,
            agent_type=agent_type,
            status=status,
        )
        GATEWAY_EXTERNAL_CALLS.inc(
            agent_type=agent_type,
            operation=operation,
            status=status,
        )


def track_credential_cache(hit: bool) -> None:
    """Track credential cache hit/miss.

    Args:
        hit: True for cache hit, False for cache miss.
    """
    if hit:
        CREDENTIAL_CACHE_HITS.inc()
    else:
        CREDENTIAL_CACHE_MISSES.inc()


def track_hybrid_verification(phase: str, duration: float) -> None:
    """Track hybrid debate verification phase timing.

    Args:
        phase: Verification phase (e.g. proposal, critique, refinement).
        duration: Duration in seconds.
    """
    HYBRID_VERIFICATION_TIME.observe(duration, phase=phase)


__all__ = [
    "GATEWAY_OPERATION_LATENCY",
    "GATEWAY_EXTERNAL_CALLS",
    "CREDENTIAL_CACHE_HITS",
    "CREDENTIAL_CACHE_MISSES",
    "HYBRID_VERIFICATION_TIME",
    "track_gateway_operation",
    "track_credential_cache",
    "track_hybrid_verification",
]
