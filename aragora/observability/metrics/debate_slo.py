"""
Debate SLI/SLO performance metrics.

Provides Prometheus metrics for tracking debate performance with
SLO-aware percentile tracking (p50, p95, p99).

Metrics exposed:
- aragora_debate_completion_duration_seconds: How long debates take (histogram)
- aragora_consensus_detection_latency_seconds: Time to detect consensus (histogram)
- aragora_agent_response_time_seconds: Per-agent response times (histogram)
- aragora_debate_success_rate: Percentage of debates reaching consensus (gauge)
- aragora_debate_success_total: Counter of successful/failed debates (counter)

Usage:
    from aragora.observability.metrics.debate_slo import (
        record_debate_completion,
        record_consensus_detection_latency,
        record_agent_response_time,
        update_debate_success_rate,
        track_debate_completion,
        track_consensus_detection,
        track_agent_response,
    )

    # Record debate completion
    record_debate_completion(duration_seconds=45.2, outcome="consensus")

    # Record consensus detection
    record_consensus_detection_latency(latency_seconds=2.5, mode="majority")

    # Record per-agent response time
    record_agent_response_time("claude", latency_seconds=3.2, phase="proposal")

    # Context managers for automatic tracking
    with track_debate_completion(outcome="consensus") as tracker:
        await arena.run()
        tracker["rounds"] = result.rounds_used

    async with track_agent_response("claude", phase="proposal"):
        response = await agent.generate(prompt)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from collections.abc import Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

__all__ = [
    # Metrics
    "DEBATE_COMPLETION_DURATION",
    "CONSENSUS_DETECTION_LATENCY",
    "AGENT_RESPONSE_TIME",
    "DEBATE_SUCCESS_RATE",
    "DEBATE_SUCCESS_TOTAL",
    # Init
    "init_debate_slo_metrics",
    # Recording functions
    "record_debate_completion_slo",
    "record_consensus_detection_latency",
    "record_agent_response_time",
    "update_debate_success_rate",
    "record_debate_outcome",
    # Context managers
    "track_debate_completion",
    "track_consensus_detection",
    "track_agent_response",
    "track_agent_response_async",
    # Stats
    "get_debate_slo_summary",
    "DebateSLOStats",
]

# =============================================================================
# Global Metric Variables
# =============================================================================

DEBATE_COMPLETION_DURATION: Any = None
CONSENSUS_DETECTION_LATENCY: Any = None
AGENT_RESPONSE_TIME: Any = None
DEBATE_SUCCESS_RATE: Any = None
DEBATE_SUCCESS_TOTAL: Any = None

_initialized = False

# Rolling window for success rate calculation
_success_window: list[tuple[float, bool]] = []  # (timestamp, success)
_success_window_seconds = 3600  # 1 hour window


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class DebateSLOStats:
    """Statistics for debate SLO metrics."""

    total_debates: int = 0
    successful_debates: int = 0
    failed_debates: int = 0
    success_rate: float = 0.0
    avg_completion_seconds: float = 0.0
    avg_consensus_latency_seconds: float = 0.0
    window_seconds: int = 3600
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# Initialization
# =============================================================================


def init_debate_slo_metrics() -> bool:
    """Initialize debate SLO metrics.

    Returns:
        True if metrics were successfully initialized
    """
    global _initialized
    global DEBATE_COMPLETION_DURATION, CONSENSUS_DETECTION_LATENCY
    global AGENT_RESPONSE_TIME, DEBATE_SUCCESS_RATE, DEBATE_SUCCESS_TOTAL

    if _initialized:
        return True

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Debate completion duration histogram
        # Buckets designed for p50/p95/p99 tracking for typical debates
        # Ranges from 1s to 20 minutes to cover quick and complex debates
        DEBATE_COMPLETION_DURATION = Histogram(
            "aragora_debate_completion_duration_seconds",
            "How long debates take to complete (seconds)",
            ["outcome"],  # consensus, timeout, error, no_consensus
            buckets=[1, 5, 10, 30, 60, 120, 180, 300, 600, 900, 1200],
        )

        # Consensus detection latency histogram
        # Measures time from voting start to consensus determination
        CONSENSUS_DETECTION_LATENCY = Histogram(
            "aragora_consensus_detection_latency_seconds",
            "Time to detect consensus (seconds)",
            ["mode"],  # none, majority, unanimous, judge, byzantine
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
        )

        # Per-agent response time histogram
        # Tracks individual model/agent performance
        AGENT_RESPONSE_TIME = Histogram(
            "aragora_agent_response_time_seconds",
            "Per-agent response time (seconds)",
            ["model", "phase"],  # model: claude, gpt-4, etc. phase: proposal, critique, vote
            buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 180],
        )

        # Debate success rate gauge (rolling window)
        DEBATE_SUCCESS_RATE = Gauge(
            "aragora_debate_success_rate",
            "Percentage of debates reaching consensus (0.0-1.0)",
        )

        # Debate outcome counter for total tracking
        DEBATE_SUCCESS_TOTAL = Counter(
            "aragora_debate_success_total",
            "Total debate outcomes by type",
            ["outcome"],  # success, failure, timeout, error
        )

        _initialized = True
        logger.debug("Debate SLO metrics initialized")
        return True

    except (ImportError, ValueError):
        logger.warning("prometheus-client not installed, debate SLO metrics disabled")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global DEBATE_COMPLETION_DURATION, CONSENSUS_DETECTION_LATENCY
    global AGENT_RESPONSE_TIME, DEBATE_SUCCESS_RATE, DEBATE_SUCCESS_TOTAL

    DEBATE_COMPLETION_DURATION = NoOpMetric()
    CONSENSUS_DETECTION_LATENCY = NoOpMetric()
    AGENT_RESPONSE_TIME = NoOpMetric()
    DEBATE_SUCCESS_RATE = NoOpMetric()
    DEBATE_SUCCESS_TOTAL = NoOpMetric()


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_debate_slo_metrics()


# =============================================================================
# Recording Functions
# =============================================================================


def record_debate_completion_slo(
    duration_seconds: float,
    outcome: str = "consensus",
) -> None:
    """Record debate completion duration for SLO tracking.

    This is the primary metric for tracking debate performance SLOs.
    The histogram enables p50/p95/p99 percentile queries in Prometheus.

    Args:
        duration_seconds: Total debate duration in seconds
        outcome: Debate outcome - one of:
            - "consensus": Reached consensus successfully
            - "no_consensus": Completed but no consensus
            - "timeout": Debate timed out
            - "error": Debate failed with error
    """
    _ensure_init()
    DEBATE_COMPLETION_DURATION.labels(outcome=outcome).observe(duration_seconds)

    # Also track in success counter
    if outcome == "consensus":
        DEBATE_SUCCESS_TOTAL.labels(outcome="success").inc()
    elif outcome == "timeout":
        DEBATE_SUCCESS_TOTAL.labels(outcome="timeout").inc()
    elif outcome == "error":
        DEBATE_SUCCESS_TOTAL.labels(outcome="error").inc()
    else:
        DEBATE_SUCCESS_TOTAL.labels(outcome="failure").inc()


def record_consensus_detection_latency(
    latency_seconds: float,
    mode: str = "majority",
) -> None:
    """Record time taken to detect/reach consensus.

    Measures the consensus phase duration, from when voting starts
    to when a consensus decision is made.

    Args:
        latency_seconds: Consensus detection time in seconds
        mode: Consensus mode used - one of:
            - "none": No consensus mode
            - "majority": Majority voting
            - "unanimous": Unanimous agreement required
            - "judge": Single judge synthesis
            - "byzantine": Byzantine fault-tolerant consensus
    """
    _ensure_init()
    CONSENSUS_DETECTION_LATENCY.labels(mode=mode).observe(latency_seconds)


def record_agent_response_time(
    model: str,
    latency_seconds: float,
    phase: str = "proposal",
) -> None:
    """Record individual agent/model response time.

    Tracks per-agent performance for identifying slow models
    and optimizing agent selection.

    Args:
        model: Model/agent name (e.g., "claude", "gpt-4", "gemini")
        latency_seconds: Response time in seconds
        phase: Debate phase - one of:
            - "proposal": Initial proposal generation
            - "critique": Critique generation
            - "revision": Proposal revision
            - "vote": Voting/evaluation
            - "synthesis": Final synthesis
    """
    _ensure_init()
    # Normalize model name to avoid cardinality explosion
    normalized_model = _normalize_model_name(model)
    AGENT_RESPONSE_TIME.labels(model=normalized_model, phase=phase).observe(latency_seconds)


def _normalize_model_name(model: str) -> str:
    """Normalize model name to control label cardinality.

    Groups similar model variants under a common name.
    """
    model_lower = model.lower()

    # Map known model families
    if "claude" in model_lower:
        if "opus" in model_lower:
            return "claude-opus"
        if "sonnet" in model_lower:
            return "claude-sonnet"
        if "haiku" in model_lower:
            return "claude-haiku"
        return "claude"
    if "gpt-4" in model_lower:
        return "gpt-4"
    if "gpt-3" in model_lower:
        return "gpt-3.5"
    if "gemini" in model_lower:
        return "gemini"
    if "grok" in model_lower:
        return "grok"
    if "mistral" in model_lower:
        return "mistral"
    if "deepseek" in model_lower:
        return "deepseek"
    if "qwen" in model_lower:
        return "qwen"
    if "llama" in model_lower:
        return "llama"

    # Truncate at first hyphen-number pattern to reduce cardinality
    parts = model.split("-")
    if len(parts) > 2:
        return "-".join(parts[:2])

    return model


def update_debate_success_rate(success: bool) -> None:
    """Update the rolling success rate gauge.

    Maintains a 1-hour rolling window of debate outcomes
    and updates the success rate gauge.

    Args:
        success: Whether the debate was successful (reached consensus)
    """
    _ensure_init()
    global _success_window

    now = time.time()
    cutoff = now - _success_window_seconds

    # Add new entry
    _success_window.append((now, success))

    # Remove old entries
    _success_window = [(ts, s) for ts, s in _success_window if ts > cutoff]

    # Calculate success rate
    if _success_window:
        successes = sum(1 for _, s in _success_window if s)
        rate = successes / len(_success_window)
        DEBATE_SUCCESS_RATE.set(rate)


def record_debate_outcome(
    duration_seconds: float,
    consensus_reached: bool,
    consensus_mode: str = "majority",
    consensus_latency_seconds: float | None = None,
) -> None:
    """Record complete debate outcome with all SLO metrics.

    Convenience function that records all relevant metrics for a completed debate.

    Args:
        duration_seconds: Total debate duration
        consensus_reached: Whether consensus was reached
        consensus_mode: Consensus mode used
        consensus_latency_seconds: Optional consensus phase duration
    """
    outcome = "consensus" if consensus_reached else "no_consensus"
    record_debate_completion_slo(duration_seconds, outcome)
    update_debate_success_rate(consensus_reached)

    if consensus_latency_seconds is not None:
        record_consensus_detection_latency(consensus_latency_seconds, consensus_mode)


# =============================================================================
# Context Managers
# =============================================================================


@contextmanager
def track_debate_completion(outcome: str = "unknown") -> Generator[dict, None, None]:
    """Context manager to track debate completion duration.

    Automatically measures and records debate duration.

    Args:
        outcome: Initial outcome (can be updated via context dict)

    Yields:
        Dict for tracking context (can set "outcome" key to override)

    Example:
        with track_debate_completion() as tracker:
            result = await arena.run()
            tracker["outcome"] = "consensus" if result.consensus_reached else "no_consensus"
            tracker["rounds"] = result.rounds_used
    """
    _ensure_init()
    start = time.perf_counter()
    ctx: dict[str, Any] = {"outcome": outcome}

    try:
        yield ctx
    except Exception as e:
        ctx["outcome"] = "error"
        ctx["error"] = str(e)
        raise
    finally:
        duration = time.perf_counter() - start
        final_outcome = ctx.get("outcome", outcome)
        record_debate_completion_slo(duration, final_outcome)
        update_debate_success_rate(final_outcome == "consensus")


@contextmanager
def track_consensus_detection(mode: str = "majority") -> Generator[None, None, None]:
    """Context manager to track consensus detection latency.

    Args:
        mode: Consensus mode being used

    Example:
        with track_consensus_detection("majority"):
            await consensus_phase.execute(ctx)
    """
    _ensure_init()
    start = time.perf_counter()

    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_consensus_detection_latency(latency, mode)


@contextmanager
def track_agent_response(model: str, phase: str = "proposal") -> Generator[None, None, None]:
    """Context manager to track agent response time (sync version).

    Args:
        model: Model/agent name
        phase: Debate phase

    Example:
        with track_agent_response("claude", "proposal"):
            response = agent.generate_sync(prompt)
    """
    _ensure_init()
    start = time.perf_counter()

    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_agent_response_time(model, latency, phase)


@asynccontextmanager
async def track_agent_response_async(model: str, phase: str = "proposal"):
    """Async context manager to track agent response time.

    Args:
        model: Model/agent name
        phase: Debate phase

    Example:
        async with track_agent_response_async("claude", "proposal"):
            response = await agent.generate(prompt)
    """
    _ensure_init()
    start = time.perf_counter()

    try:
        yield
    finally:
        latency = time.perf_counter() - start
        record_agent_response_time(model, latency, phase)


# =============================================================================
# Statistics
# =============================================================================


def get_debate_slo_summary() -> DebateSLOStats:
    """Get a summary of debate SLO metrics.

    Returns:
        DebateSLOStats with current metric values
    """
    _ensure_init()

    # Calculate stats from rolling window
    now = time.time()
    cutoff = now - _success_window_seconds

    recent = [(ts, s) for ts, s in _success_window if ts > cutoff]
    total = len(recent)
    successes = sum(1 for _, s in recent if s)
    failures = total - successes

    return DebateSLOStats(
        total_debates=total,
        successful_debates=successes,
        failed_debates=failures,
        success_rate=successes / total if total > 0 else 0.0,
        window_seconds=_success_window_seconds,
        last_updated=datetime.now().isoformat(),
    )


def reset_success_window() -> None:
    """Reset the success rate window. Primarily for testing."""
    global _success_window
    _success_window = []
