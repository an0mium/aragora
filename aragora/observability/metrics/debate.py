"""
Debate orchestration metrics.

Provides Prometheus metrics for tracking debate execution,
performance monitoring, phases, and consensus outcomes.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

# Global metric variables
DEBATE_DURATION: Any = None
DEBATE_ROUNDS: Any = None
DEBATE_PHASE_DURATION: Any = None
AGENT_PARTICIPATION: Any = None
SLOW_DEBATES_TOTAL: Any = None
SLOW_ROUNDS_TOTAL: Any = None
DEBATE_ROUND_LATENCY: Any = None
ACTIVE_DEBATES: Any = None
CONSENSUS_RATE: Any = None

_initialized = False


def init_debate_metrics() -> None:
    """Initialize debate metrics."""
    global _initialized
    global DEBATE_DURATION, DEBATE_ROUNDS, DEBATE_PHASE_DURATION
    global AGENT_PARTICIPATION, SLOW_DEBATES_TOTAL, SLOW_ROUNDS_TOTAL
    global DEBATE_ROUND_LATENCY, ACTIVE_DEBATES, CONSENSUS_RATE

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        DEBATE_DURATION = Histogram(
            "aragora_debate_duration_seconds",
            "Total debate duration in seconds",
            ["outcome"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200],
        )

        DEBATE_ROUNDS = Histogram(
            "aragora_debate_rounds_total",
            "Number of rounds per debate",
            ["outcome"],
            buckets=[1, 2, 3, 4, 5, 7, 10, 15, 20],
        )

        DEBATE_PHASE_DURATION = Histogram(
            "aragora_debate_phase_duration_seconds",
            "Duration of debate phases",
            ["phase"],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        )

        AGENT_PARTICIPATION = Counter(
            "aragora_agent_participation_total",
            "Agent participation in debates",
            ["agent", "role"],
        )

        SLOW_DEBATES_TOTAL = Counter(
            "aragora_slow_debates_total",
            "Count of debates exceeding time threshold",
            ["severity"],
        )

        SLOW_ROUNDS_TOTAL = Counter(
            "aragora_slow_rounds_total",
            "Count of rounds exceeding time threshold",
            ["phase"],
        )

        DEBATE_ROUND_LATENCY = Histogram(
            "aragora_debate_round_latency_seconds",
            "Per-round latency in debates",
            buckets=[0.5, 1, 2, 5, 10, 20, 30, 60],
        )

        ACTIVE_DEBATES = Gauge(
            "aragora_active_debates",
            "Number of currently active debates",
        )

        CONSENSUS_RATE = Gauge(
            "aragora_consensus_rate",
            "Rolling consensus achievement rate",
        )

        _initialized = True
        logger.debug("Debate metrics initialized")

    except ImportError:
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global DEBATE_DURATION, DEBATE_ROUNDS, DEBATE_PHASE_DURATION
    global AGENT_PARTICIPATION, SLOW_DEBATES_TOTAL, SLOW_ROUNDS_TOTAL
    global DEBATE_ROUND_LATENCY, ACTIVE_DEBATES, CONSENSUS_RATE

    DEBATE_DURATION = NoOpMetric()
    DEBATE_ROUNDS = NoOpMetric()
    DEBATE_PHASE_DURATION = NoOpMetric()
    AGENT_PARTICIPATION = NoOpMetric()
    SLOW_DEBATES_TOTAL = NoOpMetric()
    SLOW_ROUNDS_TOTAL = NoOpMetric()
    DEBATE_ROUND_LATENCY = NoOpMetric()
    ACTIVE_DEBATES = NoOpMetric()
    CONSENSUS_RATE = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_debate_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_debate_completion(
    duration_seconds: float,
    rounds: int,
    outcome: str,
) -> None:
    """Record debate completion metrics.

    Args:
        duration_seconds: Total debate duration
        rounds: Number of rounds in the debate
        outcome: Debate outcome (consensus, majority, timeout, etc.)
    """
    _ensure_init()
    DEBATE_DURATION.labels(outcome=outcome).observe(duration_seconds)
    DEBATE_ROUNDS.labels(outcome=outcome).observe(rounds)


def record_phase_duration(phase: str, duration_seconds: float) -> None:
    """Record duration of a debate phase.

    Args:
        phase: Phase name (proposal, critique, revision, voting, etc.)
        duration_seconds: Phase duration
    """
    _ensure_init()
    DEBATE_PHASE_DURATION.labels(phase=phase).observe(duration_seconds)


def record_agent_participation(agent: str, role: str) -> None:
    """Record agent participation in a debate.

    Args:
        agent: Agent name
        role: Role in debate (proposer, critic, judge, etc.)
    """
    _ensure_init()
    AGENT_PARTICIPATION.labels(agent=agent, role=role).inc()


def record_slow_debate(severity: str) -> None:
    """Record a slow debate.

    Args:
        severity: Severity level (warning, critical)
    """
    _ensure_init()
    SLOW_DEBATES_TOTAL.labels(severity=severity).inc()


def record_slow_round(phase: str) -> None:
    """Record a slow round.

    Args:
        phase: Phase where slowness occurred
    """
    _ensure_init()
    SLOW_ROUNDS_TOTAL.labels(phase=phase).inc()


def record_round_latency(latency_seconds: float) -> None:
    """Record per-round latency.

    Args:
        latency_seconds: Round latency in seconds
    """
    _ensure_init()
    DEBATE_ROUND_LATENCY.observe(latency_seconds)


def set_active_debates(count: int) -> None:
    """Set the number of active debates.

    Args:
        count: Number of currently active debates
    """
    _ensure_init()
    ACTIVE_DEBATES.set(count)


def increment_active_debates() -> None:
    """Increment active debates count."""
    _ensure_init()
    ACTIVE_DEBATES.inc()


def decrement_active_debates() -> None:
    """Decrement active debates count."""
    _ensure_init()
    ACTIVE_DEBATES.dec()


def set_consensus_rate(rate: float) -> None:
    """Set the consensus achievement rate.

    Args:
        rate: Consensus rate (0.0 to 1.0)
    """
    _ensure_init()
    CONSENSUS_RATE.set(rate)


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def track_debate() -> Generator[None, None, None]:
    """Context manager to track debate execution.

    Automatically increments/decrements active debates counter.

    Example:
        with track_debate():
            arena.run()
    """
    _ensure_init()
    increment_active_debates()
    try:
        yield
    finally:
        decrement_active_debates()


@contextmanager
def track_phase(phase: str) -> Generator[None, None, None]:
    """Context manager to track debate phase duration.

    Args:
        phase: Phase name

    Example:
        with track_phase("proposal"):
            await generate_proposals()
    """
    _ensure_init()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        record_phase_duration(phase, duration)


__all__ = [
    # Metrics
    "DEBATE_DURATION",
    "DEBATE_ROUNDS",
    "DEBATE_PHASE_DURATION",
    "AGENT_PARTICIPATION",
    "SLOW_DEBATES_TOTAL",
    "SLOW_ROUNDS_TOTAL",
    "DEBATE_ROUND_LATENCY",
    "ACTIVE_DEBATES",
    "CONSENSUS_RATE",
    # Functions
    "init_debate_metrics",
    "record_debate_completion",
    "record_phase_duration",
    "record_agent_participation",
    "record_slow_debate",
    "record_slow_round",
    "record_round_latency",
    "set_active_debates",
    "increment_active_debates",
    "decrement_active_debates",
    "set_consensus_rate",
    "track_debate",
    "track_phase",
]
