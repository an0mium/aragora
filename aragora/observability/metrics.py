"""
Prometheus metrics for Aragora.

Provides metrics for monitoring request rates, latencies, agent performance,
and debate statistics.

Usage:
    from aragora.observability.metrics import record_request, record_agent_call

    # Record a request
    record_request("GET", "/api/debates", 200, 0.05)

    # Record an agent call
    record_agent_call("claude", success=True, latency=1.2)

Requirements:
    pip install prometheus-client

Environment Variables:
    METRICS_ENABLED: Set to "true" to enable metrics (default: true)
    METRICS_PORT: Port for /metrics endpoint (default: 9090)

See docs/OBSERVABILITY.md for configuration guide.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar, cast

from aragora.observability.config import get_metrics_config

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Prometheus metrics - initialized lazily
_initialized = False
_metrics_server = None

# Metric instances (will be set during initialization)
REQUEST_COUNT: Any = None
REQUEST_LATENCY: Any = None
AGENT_CALLS: Any = None
AGENT_LATENCY: Any = None
ACTIVE_DEBATES: Any = None
CONSENSUS_RATE: Any = None
MEMORY_OPERATIONS: Any = None
WEBSOCKET_CONNECTIONS: Any = None
DEBATE_DURATION: Any = None
DEBATE_ROUNDS: Any = None
DEBATE_PHASE_DURATION: Any = None
AGENT_PARTICIPATION: Any = None
CACHE_HITS: Any = None
CACHE_MISSES: Any = None

# Cross-functional feature metrics
KNOWLEDGE_CACHE_HITS: Any = None
KNOWLEDGE_CACHE_MISSES: Any = None
MEMORY_COORDINATOR_WRITES: Any = None
SELECTION_FEEDBACK_ADJUSTMENTS: Any = None
WORKFLOW_TRIGGERS: Any = None
EVIDENCE_STORED: Any = None
CULTURE_PATTERNS: Any = None

# Phase 9 Cross-Pollination metrics
RLM_CACHE_HITS: Any = None
RLM_CACHE_MISSES: Any = None
CALIBRATION_ADJUSTMENTS: Any = None
LEARNING_BONUSES: Any = None
VOTING_ACCURACY_UPDATES: Any = None
ADAPTIVE_ROUND_CHANGES: Any = None


def _init_metrics() -> bool:
    """Initialize Prometheus metrics lazily."""
    global _initialized
    global REQUEST_COUNT, REQUEST_LATENCY, AGENT_CALLS, AGENT_LATENCY
    global ACTIVE_DEBATES, CONSENSUS_RATE, MEMORY_OPERATIONS, WEBSOCKET_CONNECTIONS

    if _initialized:
        return True

    config = get_metrics_config()
    if not config.enabled:
        _init_noop_metrics()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Request metrics
        REQUEST_COUNT = Counter(
            "aragora_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
        )

        REQUEST_LATENCY = Histogram(
            "aragora_request_latency_seconds",
            "HTTP request latency in seconds",
            ["endpoint"],
            buckets=config.histogram_buckets,
        )

        # Agent metrics
        AGENT_CALLS = Counter(
            "aragora_agent_calls_total",
            "Total agent API calls",
            ["agent", "status"],
        )

        AGENT_LATENCY = Histogram(
            "aragora_agent_latency_seconds",
            "Agent API call latency in seconds",
            ["agent"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        # Debate metrics
        ACTIVE_DEBATES = Gauge(
            "aragora_active_debates",
            "Number of currently active debates",
        )

        CONSENSUS_RATE = Gauge(
            "aragora_consensus_rate",
            "Rate of debates reaching consensus (0-1)",
        )

        # Memory metrics
        MEMORY_OPERATIONS = Counter(
            "aragora_memory_operations_total",
            "Total memory operations",
            ["operation", "tier"],
        )

        # WebSocket metrics
        WEBSOCKET_CONNECTIONS = Gauge(
            "aragora_websocket_connections",
            "Number of active WebSocket connections",
        )

        # Debate-specific metrics
        global DEBATE_DURATION, DEBATE_ROUNDS, DEBATE_PHASE_DURATION, AGENT_PARTICIPATION

        DEBATE_DURATION = Histogram(
            "aragora_debate_duration_seconds",
            "Debate duration in seconds",
            ["outcome"],  # consensus, no_consensus, error
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200],
        )

        DEBATE_ROUNDS = Histogram(
            "aragora_debate_rounds_total",
            "Number of rounds per debate",
            ["outcome"],
            buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )

        DEBATE_PHASE_DURATION = Histogram(
            "aragora_debate_phase_duration_seconds",
            "Duration of each debate phase",
            ["phase"],  # propose, critique, vote, consensus
            buckets=[0.5, 1, 2, 5, 10, 30, 60],
        )

        AGENT_PARTICIPATION = Counter(
            "aragora_agent_participation_total",
            "Agent participation in debates",
            ["agent", "phase"],
        )

        # Cache metrics
        global CACHE_HITS, CACHE_MISSES

        CACHE_HITS = Counter(
            "aragora_cache_hits_total",
            "Cache hit count",
            ["cache_name"],
        )

        CACHE_MISSES = Counter(
            "aragora_cache_misses_total",
            "Cache miss count",
            ["cache_name"],
        )

        # Cross-functional feature metrics
        global KNOWLEDGE_CACHE_HITS, KNOWLEDGE_CACHE_MISSES
        global MEMORY_COORDINATOR_WRITES, SELECTION_FEEDBACK_ADJUSTMENTS
        global WORKFLOW_TRIGGERS, EVIDENCE_STORED, CULTURE_PATTERNS

        KNOWLEDGE_CACHE_HITS = Counter(
            "aragora_knowledge_cache_hits_total",
            "Knowledge query cache hits",
        )

        KNOWLEDGE_CACHE_MISSES = Counter(
            "aragora_knowledge_cache_misses_total",
            "Knowledge query cache misses",
        )

        MEMORY_COORDINATOR_WRITES = Counter(
            "aragora_memory_coordinator_writes_total",
            "Atomic memory coordinator writes",
            ["status"],  # success, failed, rolled_back
        )

        SELECTION_FEEDBACK_ADJUSTMENTS = Counter(
            "aragora_selection_feedback_adjustments_total",
            "Agent selection weight adjustments",
            ["agent", "direction"],  # up, down
        )

        WORKFLOW_TRIGGERS = Counter(
            "aragora_workflow_triggers_total",
            "Post-debate workflow triggers",
            ["status"],  # triggered, skipped, completed, failed
        )

        EVIDENCE_STORED = Counter(
            "aragora_evidence_stored_total",
            "Evidence items stored in knowledge mound",
        )

        CULTURE_PATTERNS = Counter(
            "aragora_culture_patterns_total",
            "Culture patterns extracted from debates",
        )

        # Phase 9 Cross-Pollination metrics
        global RLM_CACHE_HITS, RLM_CACHE_MISSES
        global CALIBRATION_ADJUSTMENTS, LEARNING_BONUSES
        global VOTING_ACCURACY_UPDATES, ADAPTIVE_ROUND_CHANGES

        RLM_CACHE_HITS = Counter(
            "aragora_rlm_cache_hits_total",
            "RLM compression cache hits",
        )

        RLM_CACHE_MISSES = Counter(
            "aragora_rlm_cache_misses_total",
            "RLM compression cache misses",
        )

        CALIBRATION_ADJUSTMENTS = Counter(
            "aragora_calibration_adjustments_total",
            "Proposal confidence calibrations applied",
            ["agent"],
        )

        LEARNING_BONUSES = Counter(
            "aragora_learning_bonuses_total",
            "Learning efficiency ELO bonuses applied",
            ["agent", "category"],  # rapid, steady, slow
        )

        VOTING_ACCURACY_UPDATES = Counter(
            "aragora_voting_accuracy_updates_total",
            "Voting accuracy records updated",
            ["result"],  # correct, incorrect
        )

        ADAPTIVE_ROUND_CHANGES = Counter(
            "aragora_adaptive_round_changes_total",
            "Debate round count adjustments from memory strategy",
            ["direction"],  # increased, decreased, unchanged
        )

        _initialized = True
        logger.info("Prometheus metrics initialized")
        return True

    except ImportError as e:
        logger.warning(
            f"prometheus-client not installed, metrics disabled: {e}. "
            "Install with: pip install prometheus-client"
        )
        _init_noop_metrics()
        _initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")
        _init_noop_metrics()
        _initialized = True
        return False


def _init_noop_metrics() -> None:
    """Initialize no-op metrics for when prometheus is disabled."""
    global REQUEST_COUNT, REQUEST_LATENCY, AGENT_CALLS, AGENT_LATENCY
    global ACTIVE_DEBATES, CONSENSUS_RATE, MEMORY_OPERATIONS, WEBSOCKET_CONNECTIONS
    global DEBATE_DURATION, DEBATE_ROUNDS, DEBATE_PHASE_DURATION, AGENT_PARTICIPATION
    global CACHE_HITS, CACHE_MISSES
    global KNOWLEDGE_CACHE_HITS, KNOWLEDGE_CACHE_MISSES
    global MEMORY_COORDINATOR_WRITES, SELECTION_FEEDBACK_ADJUSTMENTS
    global WORKFLOW_TRIGGERS, EVIDENCE_STORED, CULTURE_PATTERNS
    global RLM_CACHE_HITS, RLM_CACHE_MISSES
    global CALIBRATION_ADJUSTMENTS, LEARNING_BONUSES
    global VOTING_ACCURACY_UPDATES, ADAPTIVE_ROUND_CHANGES

    class NoOpMetric:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpMetric":
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def dec(self, amount: float = 1) -> None:
            pass

        def set(self, value: float) -> None:
            pass

        def observe(self, value: float) -> None:
            pass

    REQUEST_COUNT = NoOpMetric()
    REQUEST_LATENCY = NoOpMetric()
    AGENT_CALLS = NoOpMetric()
    AGENT_LATENCY = NoOpMetric()
    ACTIVE_DEBATES = NoOpMetric()
    CONSENSUS_RATE = NoOpMetric()
    MEMORY_OPERATIONS = NoOpMetric()
    WEBSOCKET_CONNECTIONS = NoOpMetric()
    DEBATE_DURATION = NoOpMetric()
    DEBATE_ROUNDS = NoOpMetric()
    DEBATE_PHASE_DURATION = NoOpMetric()
    AGENT_PARTICIPATION = NoOpMetric()
    CACHE_HITS = NoOpMetric()
    CACHE_MISSES = NoOpMetric()
    KNOWLEDGE_CACHE_HITS = NoOpMetric()
    KNOWLEDGE_CACHE_MISSES = NoOpMetric()
    MEMORY_COORDINATOR_WRITES = NoOpMetric()
    SELECTION_FEEDBACK_ADJUSTMENTS = NoOpMetric()
    WORKFLOW_TRIGGERS = NoOpMetric()
    EVIDENCE_STORED = NoOpMetric()
    CULTURE_PATTERNS = NoOpMetric()
    RLM_CACHE_HITS = NoOpMetric()
    RLM_CACHE_MISSES = NoOpMetric()
    CALIBRATION_ADJUSTMENTS = NoOpMetric()
    LEARNING_BONUSES = NoOpMetric()
    VOTING_ACCURACY_UPDATES = NoOpMetric()
    ADAPTIVE_ROUND_CHANGES = NoOpMetric()


def start_metrics_server() -> Optional[Any]:
    """Start the Prometheus metrics HTTP server.

    Returns:
        The server instance, or None if metrics disabled
    """
    global _metrics_server

    if not _init_metrics():
        return None

    if _metrics_server is not None:
        return _metrics_server

    config = get_metrics_config()
    if not config.enabled:
        return None

    try:
        from prometheus_client import start_http_server

        _metrics_server = start_http_server(config.port)
        logger.info(f"Prometheus metrics server started on port {config.port}")
        return _metrics_server
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        return None


def record_request(
    method: str,
    endpoint: str,
    status: int,
    latency: float,
) -> None:
    """Record an HTTP request metric.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: Request endpoint path
        status: HTTP status code
        latency: Request latency in seconds
    """
    _init_metrics()

    # Normalize endpoint for cardinality control
    normalized_endpoint = _normalize_endpoint(endpoint)

    REQUEST_COUNT.labels(method=method, endpoint=normalized_endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(endpoint=normalized_endpoint).observe(latency)


def record_agent_call(
    agent: str,
    success: bool,
    latency: float,
) -> None:
    """Record an agent API call metric.

    Args:
        agent: Agent name
        success: Whether the call succeeded
        latency: Call latency in seconds
    """
    _init_metrics()

    status = "success" if success else "error"
    AGENT_CALLS.labels(agent=agent, status=status).inc()
    AGENT_LATENCY.labels(agent=agent).observe(latency)


@contextmanager
def track_debate() -> Generator[None, None, None]:
    """Context manager to track active debates.

    Example:
        with track_debate():
            # Debate is running
            await arena.run()
    """
    _init_metrics()

    ACTIVE_DEBATES.inc()
    try:
        yield
    finally:
        ACTIVE_DEBATES.dec()


def set_consensus_rate(rate: float) -> None:
    """Set the consensus rate metric.

    Args:
        rate: Consensus rate between 0 and 1
    """
    _init_metrics()
    CONSENSUS_RATE.set(rate)


def record_memory_operation(operation: str, tier: str) -> None:
    """Record a memory operation.

    Args:
        operation: Operation type (store, query, promote, demote)
        tier: Memory tier (fast, medium, slow, glacial)
    """
    _init_metrics()
    MEMORY_OPERATIONS.labels(operation=operation, tier=tier).inc()


def track_websocket_connection(connected: bool) -> None:
    """Track WebSocket connection state.

    Args:
        connected: True if connected, False if disconnected
    """
    _init_metrics()
    if connected:
        WEBSOCKET_CONNECTIONS.inc()
    else:
        WEBSOCKET_CONNECTIONS.dec()


def measure_latency(metric_name: str = "request") -> Callable[[F], F]:
    """Decorator to measure function latency.

    Args:
        metric_name: Name for the latency metric

    Returns:
        Decorated function with latency measurement
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _init_metrics()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=metric_name).observe(latency)

        return cast(F, wrapper)

    return decorator


def measure_async_latency(metric_name: str = "request") -> Callable[[F], F]:
    """Decorator to measure async function latency.

    Args:
        metric_name: Name for the latency metric

    Returns:
        Decorated async function with latency measurement
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            _init_metrics()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                latency = time.perf_counter() - start
                REQUEST_LATENCY.labels(endpoint=metric_name).observe(latency)

        return cast(F, wrapper)

    return decorator


def _normalize_endpoint(endpoint: str) -> str:
    """Normalize endpoint path to control cardinality.

    Replaces dynamic path segments (IDs, UUIDs) with placeholders.

    Args:
        endpoint: Raw endpoint path

    Returns:
        Normalized endpoint path
    """
    import re

    # Replace UUIDs
    endpoint = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        ":id",
        endpoint,
        flags=re.IGNORECASE,
    )

    # Replace numeric IDs
    endpoint = re.sub(r"/\d+", "/:id", endpoint)

    # Replace base64-like tokens
    endpoint = re.sub(r"/[A-Za-z0-9_-]{20,}", "/:token", endpoint)

    return endpoint


# =============================================================================
# Debate-Specific Metrics
# =============================================================================


def record_debate_completion(
    duration_seconds: float,
    rounds: int,
    outcome: str,
) -> None:
    """Record metrics when a debate completes.

    Args:
        duration_seconds: Total debate duration in seconds
        rounds: Number of rounds completed
        outcome: Debate outcome ("consensus", "no_consensus", "error")
    """
    _init_metrics()
    DEBATE_DURATION.labels(outcome=outcome).observe(duration_seconds)
    DEBATE_ROUNDS.labels(outcome=outcome).observe(rounds)


def record_phase_duration(phase: str, duration_seconds: float) -> None:
    """Record the duration of a debate phase.

    Args:
        phase: Phase name ("propose", "critique", "vote", "consensus")
        duration_seconds: Phase duration in seconds
    """
    _init_metrics()
    DEBATE_PHASE_DURATION.labels(phase=phase).observe(duration_seconds)


def record_agent_participation(agent: str, phase: str) -> None:
    """Record agent participation in a debate phase.

    Args:
        agent: Agent name
        phase: Phase name
    """
    _init_metrics()
    AGENT_PARTICIPATION.labels(agent=agent, phase=phase).inc()


@contextmanager
def track_phase(phase: str) -> Generator[None, None, None]:
    """Context manager to track phase duration.

    Args:
        phase: Phase name

    Example:
        with track_phase("propose"):
            # Phase is running
            await run_propose_phase()
    """
    _init_metrics()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        DEBATE_PHASE_DURATION.labels(phase=phase).observe(duration)


# =============================================================================
# Cache Metrics
# =============================================================================


def record_cache_hit(cache_name: str) -> None:
    """Record a cache hit.

    Args:
        cache_name: Name of the cache
    """
    _init_metrics()
    CACHE_HITS.labels(cache_name=cache_name).inc()


def record_cache_miss(cache_name: str) -> None:
    """Record a cache miss.

    Args:
        cache_name: Name of the cache
    """
    _init_metrics()
    CACHE_MISSES.labels(cache_name=cache_name).inc()


# Cross-functional feature metrics helpers


def record_knowledge_cache_hit() -> None:
    """Record a knowledge query cache hit."""
    _init_metrics()
    KNOWLEDGE_CACHE_HITS.inc()


def record_knowledge_cache_miss() -> None:
    """Record a knowledge query cache miss."""
    _init_metrics()
    KNOWLEDGE_CACHE_MISSES.inc()


def record_memory_coordinator_write(status: str) -> None:
    """Record a memory coordinator write operation.

    Args:
        status: Write status (success, failed, rolled_back)
    """
    _init_metrics()
    MEMORY_COORDINATOR_WRITES.labels(status=status).inc()


def record_selection_feedback_adjustment(agent: str, direction: str) -> None:
    """Record an agent selection weight adjustment.

    Args:
        agent: Agent name
        direction: Adjustment direction (up, down)
    """
    _init_metrics()
    SELECTION_FEEDBACK_ADJUSTMENTS.labels(agent=agent, direction=direction).inc()


def record_workflow_trigger(status: str) -> None:
    """Record a post-debate workflow trigger.

    Args:
        status: Trigger status (triggered, skipped, completed, failed)
    """
    _init_metrics()
    WORKFLOW_TRIGGERS.labels(status=status).inc()


def record_evidence_stored(count: int = 1) -> None:
    """Record evidence items stored in knowledge mound.

    Args:
        count: Number of evidence items stored
    """
    _init_metrics()
    EVIDENCE_STORED.inc(count)


def record_culture_patterns(count: int = 1) -> None:
    """Record culture patterns extracted from debates.

    Args:
        count: Number of patterns extracted
    """
    _init_metrics()
    CULTURE_PATTERNS.inc(count)


# Phase 9 Cross-Pollination metrics helpers


def record_rlm_cache_hit() -> None:
    """Record an RLM compression cache hit."""
    _init_metrics()
    RLM_CACHE_HITS.inc()


def record_rlm_cache_miss() -> None:
    """Record an RLM compression cache miss."""
    _init_metrics()
    RLM_CACHE_MISSES.inc()


def record_calibration_adjustment(agent: str) -> None:
    """Record a proposal confidence calibration adjustment.

    Args:
        agent: Agent name whose confidence was calibrated
    """
    _init_metrics()
    CALIBRATION_ADJUSTMENTS.labels(agent=agent).inc()


def record_learning_bonus(agent: str, category: str) -> None:
    """Record a learning efficiency ELO bonus.

    Args:
        agent: Agent name
        category: Learning category (rapid, steady, slow)
    """
    _init_metrics()
    LEARNING_BONUSES.labels(agent=agent, category=category).inc()


def record_voting_accuracy_update(result: str) -> None:
    """Record a voting accuracy update.

    Args:
        result: Vote result (correct, incorrect)
    """
    _init_metrics()
    VOTING_ACCURACY_UPDATES.labels(result=result).inc()


def record_adaptive_round_change(direction: str) -> None:
    """Record a debate round count adjustment.

    Args:
        direction: Change direction (increased, decreased, unchanged)
    """
    _init_metrics()
    ADAPTIVE_ROUND_CHANGES.labels(direction=direction).inc()
