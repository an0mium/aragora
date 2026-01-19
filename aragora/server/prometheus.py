"""
Prometheus metrics for Aragora server.

Provides OpenMetrics-compliant metrics for monitoring:
- Debate operations (latency, token usage, outcomes)
- Agent performance (generation time, failures)
- HTTP request metrics (latency per endpoint)
- WebSocket connections
- Rate limiter state
- Cache statistics
"""

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Dict

# Try to import prometheus_client, fall back to simple implementation
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


# ============================================================================
# Metric Definitions (when prometheus_client is available)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Debate metrics
    DEBATE_DURATION = Histogram(
        "aragora_debate_duration_seconds",
        "Time spent in debate execution",
        ["outcome", "agent_count"],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600],
    )

    DEBATE_ROUNDS = Histogram(
        "aragora_debate_rounds_total",
        "Number of rounds per debate",
        ["outcome"],
        buckets=[1, 2, 3, 4, 5, 7, 10],
    )

    DEBATE_TOKENS = Counter(
        "aragora_debate_tokens_total",
        "Total tokens used in debates",
        ["model", "direction"],  # direction: input/output
    )

    DEBATES_TOTAL = Counter(
        "aragora_debates_total",
        "Total number of debates",
        ["outcome"],  # consensus, no_consensus, error, timeout
    )

    # Agent metrics
    AGENT_GENERATION_DURATION = Histogram(
        "aragora_agent_generation_seconds",
        "Time spent generating agent responses",
        ["agent_type", "model"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60],
    )

    AGENT_FAILURES = Counter(
        "aragora_agent_failures_total",
        "Total agent failures",
        ["agent_type", "error_type"],
    )

    AGENT_CIRCUIT_BREAKER = Gauge(
        "aragora_agent_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        ["agent_type"],
    )

    # HTTP metrics
    HTTP_REQUEST_DURATION = Histogram(
        "aragora_http_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint", "status"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
    )

    HTTP_REQUESTS_TOTAL = Counter(
        "aragora_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )

    # WebSocket metrics
    WEBSOCKET_CONNECTIONS = Gauge(
        "aragora_websocket_connections_active",
        "Number of active WebSocket connections",
    )

    WEBSOCKET_MESSAGES = Counter(
        "aragora_websocket_messages_total",
        "Total WebSocket messages",
        ["direction", "message_type"],  # direction: sent/received
    )

    # Rate limiter metrics
    RATE_LIMIT_HITS = Counter(
        "aragora_rate_limit_hits_total",
        "Rate limit hits",
        ["limit_type"],  # token, ip
    )

    RATE_LIMIT_TOKENS_TRACKED = Gauge(
        "aragora_rate_limit_tokens_tracked",
        "Number of tokens being rate-limited",
    )

    # Cache metrics
    CACHE_SIZE = Gauge(
        "aragora_cache_size_entries",
        "Number of entries in cache",
        ["cache_name"],
    )

    CACHE_HITS = Counter(
        "aragora_cache_hits_total",
        "Cache hits",
        ["cache_name"],
    )

    CACHE_MISSES = Counter(
        "aragora_cache_misses_total",
        "Cache misses",
        ["cache_name"],
    )

    # System info
    ARAGORA_INFO = Info(
        "aragora",
        "Aragora server information",
    )

    # Database metrics
    DB_QUERY_DURATION = Histogram(
        "aragora_db_query_duration_seconds",
        "Database query execution time",
        ["operation", "table"],  # operation: select, insert, update, delete
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5],
    )

    DB_QUERY_TOTAL = Counter(
        "aragora_db_queries_total",
        "Total database queries",
        ["operation", "table"],
    )

    DB_ERRORS_TOTAL = Counter(
        "aragora_db_errors_total",
        "Total database errors",
        ["error_type", "operation"],
    )

    DB_CONNECTION_POOL_SIZE = Gauge(
        "aragora_db_connection_pool_size",
        "Database connection pool size",
        ["state"],  # active, idle
    )

    # Memory tier metrics
    MEMORY_TIER_SIZE = Gauge(
        "aragora_memory_tier_size",
        "Number of memories in each tier",
        ["tier"],  # fast, medium, slow, glacial
    )

    MEMORY_TIER_TRANSITIONS = Counter(
        "aragora_memory_tier_transitions_total",
        "Memory tier transitions",
        ["from_tier", "to_tier"],  # e.g., fast->medium, medium->slow
    )

    MEMORY_OPERATIONS = Counter(
        "aragora_memory_operations_total",
        "Memory operations by type",
        ["operation"],  # store, retrieve, consolidate, prune
    )

    # Nomic loop phase metrics
    NOMIC_PHASE_DURATION = Histogram(
        "aragora_nomic_phase_duration_seconds",
        "Time spent in each nomic loop phase",
        ["phase", "outcome"],  # phase: context, debate, design, implement, verify, commit
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200],
    )

    NOMIC_PHASE_TOTAL = Counter(
        "aragora_nomic_phases_total",
        "Total nomic phases executed",
        ["phase", "outcome"],  # outcome: success, failure, skipped
    )

    NOMIC_CYCLE_DURATION = Histogram(
        "aragora_nomic_cycle_duration_seconds",
        "Total time for a complete nomic cycle",
        ["outcome"],  # outcome: success, failure, partial
        buckets=[60, 120, 300, 600, 1200, 1800, 3600],
    )

    NOMIC_CYCLE_TOTAL = Counter(
        "aragora_nomic_cycles_total",
        "Total nomic cycles executed",
        ["outcome"],
    )

    NOMIC_AGENT_PHASE_DURATION = Histogram(
        "aragora_nomic_agent_phase_seconds",
        "Time spent by each agent in a phase",
        ["phase", "agent"],
        buckets=[1, 5, 10, 30, 60, 120, 300],
    )

    # Control Plane metrics
    CONTROL_PLANE_TASKS_TOTAL = Counter(
        "aragora_control_plane_tasks_total",
        "Total tasks submitted to the control plane",
        ["task_type", "priority"],
    )

    CONTROL_PLANE_TASK_STATUS = Gauge(
        "aragora_control_plane_task_status_count",
        "Number of tasks by status",
        ["status"],  # pending, running, completed, failed, cancelled
    )

    CONTROL_PLANE_TASK_DURATION = Histogram(
        "aragora_control_plane_task_duration_seconds",
        "Task execution duration",
        ["task_type", "outcome"],  # outcome: completed, failed, timeout
        buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
    )

    CONTROL_PLANE_QUEUE_DEPTH = Gauge(
        "aragora_control_plane_queue_depth",
        "Number of tasks in queue by priority",
        ["priority"],  # low, normal, high, urgent
    )

    CONTROL_PLANE_AGENTS_REGISTERED = Gauge(
        "aragora_control_plane_agents_registered",
        "Number of registered agents",
        ["status"],  # available, busy, offline
    )

    CONTROL_PLANE_AGENT_HEALTH = Gauge(
        "aragora_control_plane_agent_health",
        "Agent health status (0=unhealthy, 1=degraded, 2=healthy)",
        ["agent_id"],
    )

    CONTROL_PLANE_AGENT_LATENCY = Histogram(
        "aragora_control_plane_agent_latency_seconds",
        "Agent health check latency",
        ["agent_id"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
    )

    CONTROL_PLANE_TASK_RETRIES = Counter(
        "aragora_control_plane_task_retries_total",
        "Total task retries",
        ["task_type", "reason"],  # reason: timeout, error, capability_mismatch
    )

    CONTROL_PLANE_DEAD_LETTER_QUEUE = Gauge(
        "aragora_control_plane_dead_letter_queue_size",
        "Number of tasks in dead letter queue",
    )

    CONTROL_PLANE_CLAIM_LATENCY = Histogram(
        "aragora_control_plane_claim_latency_seconds",
        "Time to claim a task from queue",
        ["priority"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
    )

    # RLM (Recursive Language Models) metrics
    RLM_COMPRESSIONS = Counter(
        "aragora_rlm_compressions_total",
        "Total RLM compression operations",
        ["source_type", "status"],
    )

    RLM_COMPRESSION_RATIO = Histogram(
        "aragora_rlm_compression_ratio",
        "Compression ratio (compressed/original tokens)",
        ["source_type"],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    RLM_TOKENS_SAVED = Counter(
        "aragora_rlm_tokens_saved_total",
        "Total tokens saved through compression",
        ["source_type"],
    )

    RLM_COMPRESSION_DURATION = Histogram(
        "aragora_rlm_compression_duration_seconds",
        "Time taken for compression operations",
        ["source_type", "levels"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    RLM_QUERIES = Counter(
        "aragora_rlm_queries_total",
        "Total RLM context queries",
        ["query_type", "level"],
    )

    RLM_QUERY_DURATION = Histogram(
        "aragora_rlm_query_duration_seconds",
        "Time taken for context queries",
        ["query_type"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )

    RLM_CACHE_HITS = Counter(
        "aragora_rlm_cache_hits_total",
        "RLM compression cache hits",
    )

    RLM_CACHE_MISSES = Counter(
        "aragora_rlm_cache_misses_total",
        "RLM compression cache misses",
    )

    RLM_CONTEXT_LEVELS = Histogram(
        "aragora_rlm_context_levels",
        "Number of abstraction levels created",
        ["source_type"],
        buckets=[1, 2, 3, 4, 5],
    )

    RLM_MEMORY_USAGE = Gauge(
        "aragora_rlm_memory_bytes",
        "Memory used by RLM context cache",
    )

    # RLM Iterative refinement metrics (Prime Intellect alignment)
    RLM_REFINEMENT_ITERATIONS = Histogram(
        "aragora_rlm_refinement_iterations",
        "Number of refinement iterations until ready=True",
        ["strategy"],
        buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )

    RLM_REFINEMENT_SUCCESS = Counter(
        "aragora_rlm_refinement_success_total",
        "Successful refinements (ready=True before max iterations)",
        ["strategy"],
    )

    RLM_REFINEMENT_DURATION = Histogram(
        "aragora_rlm_refinement_duration_seconds",
        "Total time for refinement loop (all iterations)",
        ["strategy"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    RLM_READY_FALSE_RATE = Counter(
        "aragora_rlm_ready_false_total",
        "Times LLM signaled ready=False (needs refinement)",
        ["iteration"],
    )


# ============================================================================
# Fallback Implementation (when prometheus_client not available)
# ============================================================================


@dataclass
class SimpleMetrics:
    """Simple metrics storage when prometheus_client is not available."""

    counters: Dict[str, float] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
    histograms: Dict[str, list] = field(default_factory=dict)
    info: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def inc_counter(
        self, name: str, labels: Dict[str, str] | None = None, value: float = 1
    ) -> None:
        key = self._make_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] | None = None) -> None:
        key = self._make_key(name, labels)
        self.gauges[key] = value

    def observe_histogram(
        self, name: str, value: float, labels: Dict[str, str] | None = None
    ) -> None:
        key = self._make_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)

    def set_info(self, name: str, info: Dict[str, str]) -> None:
        self.info[name] = info

    def _make_key(self, name: str, labels: Dict[str, str] | None = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def generate_output(self) -> str:
        """Generate Prometheus-format output."""
        lines = []

        # Counters
        for key, value in sorted(self.counters.items()):
            lines.append(f"{key} {value}")

        # Gauges
        for key, value in sorted(self.gauges.items()):
            lines.append(f"{key} {value}")

        # Histograms (simplified - just count and sum)
        for key, values in sorted(self.histograms.items()):
            if values:
                lines.append(f"{key}_count {len(values)}")
                lines.append(f"{key}_sum {sum(values)}")

        # Info
        for name, info in self.info.items():
            info_str = ",".join(f'{k}="{v}"' for k, v in info.items())
            lines.append(f"{name}_info{{{info_str}}} 1")

        return "\n".join(lines) + "\n"


# Global simple metrics instance (fallback)
_simple_metrics = SimpleMetrics()


# ============================================================================
# Public API
# ============================================================================


def get_metrics_output() -> tuple[str, str]:
    """
    Get metrics in Prometheus format.

    Returns:
        Tuple of (content, content_type)
    """
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY).decode("utf-8"), CONTENT_TYPE_LATEST
    else:
        return _simple_metrics.generate_output(), CONTENT_TYPE_LATEST


def is_prometheus_available() -> bool:
    """Check if prometheus_client is installed."""
    return PROMETHEUS_AVAILABLE


def get_prometheus_metrics() -> str:
    """Get metrics text in Prometheus format."""
    content, _ = get_metrics_output()
    return content


# ============================================================================
# Recording Functions
# ============================================================================


def record_debate_completed(
    duration_seconds: float,
    rounds_used: int,
    outcome: str,  # "consensus", "no_consensus", "error", "timeout"
    agent_count: int,
) -> None:
    """Record a completed debate."""
    if PROMETHEUS_AVAILABLE:
        DEBATE_DURATION.labels(outcome=outcome, agent_count=str(agent_count)).observe(
            duration_seconds
        )
        DEBATE_ROUNDS.labels(outcome=outcome).observe(rounds_used)
        DEBATES_TOTAL.labels(outcome=outcome).inc()
    else:
        _simple_metrics.observe_histogram(
            "aragora_debate_duration_seconds",
            duration_seconds,
            {"outcome": outcome, "agent_count": str(agent_count)},
        )
        _simple_metrics.inc_counter("aragora_debates_total", {"outcome": outcome})


def record_tokens_used(model: str, input_tokens: int, output_tokens: int) -> None:
    """Record token usage."""
    if PROMETHEUS_AVAILABLE:
        DEBATE_TOKENS.labels(model=model, direction="input").inc(input_tokens)
        DEBATE_TOKENS.labels(model=model, direction="output").inc(output_tokens)
    else:
        _simple_metrics.inc_counter(
            "aragora_debate_tokens_total",
            {"model": model, "direction": "input"},
            input_tokens,
        )
        _simple_metrics.inc_counter(
            "aragora_debate_tokens_total",
            {"model": model, "direction": "output"},
            output_tokens,
        )


def record_agent_generation(agent_type: str, model: str, duration_seconds: float) -> None:
    """Record agent generation time."""
    if PROMETHEUS_AVAILABLE:
        AGENT_GENERATION_DURATION.labels(agent_type=agent_type, model=model).observe(
            duration_seconds
        )
    else:
        _simple_metrics.observe_histogram(
            "aragora_agent_generation_seconds",
            duration_seconds,
            {"agent_type": agent_type, "model": model},
        )


def record_agent_failure(agent_type: str, error_type: str) -> None:
    """Record an agent failure."""
    if PROMETHEUS_AVAILABLE:
        AGENT_FAILURES.labels(agent_type=agent_type, error_type=error_type).inc()
    else:
        _simple_metrics.inc_counter(
            "aragora_agent_failures_total",
            {"agent_type": agent_type, "error_type": error_type},
        )


def set_circuit_breaker_state(agent_type: str, state: int) -> None:
    """Set circuit breaker state (0=closed, 1=open, 2=half-open)."""
    if PROMETHEUS_AVAILABLE:
        AGENT_CIRCUIT_BREAKER.labels(agent_type=agent_type).set(state)
    else:
        _simple_metrics.set_gauge(
            "aragora_agent_circuit_breaker_state",
            state,
            {"agent_type": agent_type},
        )


def record_http_request(method: str, endpoint: str, status: int, duration_seconds: float) -> None:
    """Record an HTTP request."""
    if PROMETHEUS_AVAILABLE:
        HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint, status=str(status)).observe(
            duration_seconds
        )
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    else:
        _simple_metrics.observe_histogram(
            "aragora_http_request_duration_seconds",
            duration_seconds,
            {"method": method, "endpoint": endpoint, "status": str(status)},
        )
        _simple_metrics.inc_counter(
            "aragora_http_requests_total",
            {"method": method, "endpoint": endpoint, "status": str(status)},
        )


def set_websocket_connections(count: int) -> None:
    """Set active WebSocket connection count."""
    if PROMETHEUS_AVAILABLE:
        WEBSOCKET_CONNECTIONS.set(count)
    else:
        _simple_metrics.set_gauge("aragora_websocket_connections_active", count)


def record_websocket_message(direction: str, message_type: str) -> None:
    """Record a WebSocket message."""
    if PROMETHEUS_AVAILABLE:
        WEBSOCKET_MESSAGES.labels(direction=direction, message_type=message_type).inc()
    else:
        _simple_metrics.inc_counter(
            "aragora_websocket_messages_total",
            {"direction": direction, "message_type": message_type},
        )


def record_rate_limit_hit(limit_type: str) -> None:
    """Record a rate limit hit."""
    if PROMETHEUS_AVAILABLE:
        RATE_LIMIT_HITS.labels(limit_type=limit_type).inc()
    else:
        _simple_metrics.inc_counter("aragora_rate_limit_hits_total", {"limit_type": limit_type})


def set_rate_limit_tokens_tracked(count: int) -> None:
    """Set number of tokens being tracked for rate limiting."""
    if PROMETHEUS_AVAILABLE:
        RATE_LIMIT_TOKENS_TRACKED.set(count)
    else:
        _simple_metrics.set_gauge("aragora_rate_limit_tokens_tracked", count)


def set_cache_size(cache_name: str, size: int) -> None:
    """Set cache size."""
    if PROMETHEUS_AVAILABLE:
        CACHE_SIZE.labels(cache_name=cache_name).set(size)
    else:
        _simple_metrics.set_gauge("aragora_cache_size_entries", size, {"cache_name": cache_name})


def record_cache_hit(cache_name: str) -> None:
    """Record a cache hit."""
    if PROMETHEUS_AVAILABLE:
        CACHE_HITS.labels(cache_name=cache_name).inc()
    else:
        _simple_metrics.inc_counter("aragora_cache_hits_total", {"cache_name": cache_name})


def record_cache_miss(cache_name: str) -> None:
    """Record a cache miss."""
    if PROMETHEUS_AVAILABLE:
        CACHE_MISSES.labels(cache_name=cache_name).inc()
    else:
        _simple_metrics.inc_counter("aragora_cache_misses_total", {"cache_name": cache_name})


def set_server_info(version: str, python_version: str, start_time: float) -> None:
    """Set server information."""
    if PROMETHEUS_AVAILABLE:
        ARAGORA_INFO.info(
            {
                "version": version,
                "python_version": python_version,
                "start_time": str(int(start_time)),
            }
        )
    else:
        _simple_metrics.set_info(
            "aragora",
            {
                "version": version,
                "python_version": python_version,
                "start_time": str(int(start_time)),
            },
        )


def record_db_query(operation: str, table: str, duration_seconds: float) -> None:
    """Record a database query.

    Args:
        operation: Query operation type (select, insert, update, delete)
        table: Table name being queried
        duration_seconds: Query execution time
    """
    if PROMETHEUS_AVAILABLE:
        DB_QUERY_DURATION.labels(operation=operation, table=table).observe(duration_seconds)
        DB_QUERY_TOTAL.labels(operation=operation, table=table).inc()
    else:
        _simple_metrics.observe_histogram(
            "aragora_db_query_duration_seconds",
            duration_seconds,
            {"operation": operation, "table": table},
        )
        _simple_metrics.inc_counter(
            "aragora_db_queries_total",
            {"operation": operation, "table": table},
        )


def record_db_error(error_type: str, operation: str) -> None:
    """Record a database error.

    Args:
        error_type: Type of error (e.g., "timeout", "connection", "constraint")
        operation: Operation that failed
    """
    if PROMETHEUS_AVAILABLE:
        DB_ERRORS_TOTAL.labels(error_type=error_type, operation=operation).inc()
    else:
        _simple_metrics.inc_counter(
            "aragora_db_errors_total",
            {"error_type": error_type, "operation": operation},
        )


def set_db_pool_size(active: int, idle: int) -> None:
    """Set database connection pool sizes.

    Args:
        active: Number of active connections
        idle: Number of idle connections
    """
    if PROMETHEUS_AVAILABLE:
        DB_CONNECTION_POOL_SIZE.labels(state="active").set(active)
        DB_CONNECTION_POOL_SIZE.labels(state="idle").set(idle)
    else:
        _simple_metrics.set_gauge("aragora_db_connection_pool_size", active, {"state": "active"})
        _simple_metrics.set_gauge("aragora_db_connection_pool_size", idle, {"state": "idle"})


def set_memory_tier_size(tier: str, size: int) -> None:
    """Set the number of memories in a tier.

    Args:
        tier: Memory tier name (fast, medium, slow, glacial)
        size: Number of memories in the tier
    """
    if PROMETHEUS_AVAILABLE:
        MEMORY_TIER_SIZE.labels(tier=tier).set(size)
    else:
        _simple_metrics.set_gauge("aragora_memory_tier_size", size, {"tier": tier})


def record_memory_tier_transition(from_tier: str, to_tier: str) -> None:
    """Record a memory tier transition.

    Args:
        from_tier: Source tier (fast, medium, slow)
        to_tier: Destination tier (medium, slow, glacial)
    """
    if PROMETHEUS_AVAILABLE:
        MEMORY_TIER_TRANSITIONS.labels(from_tier=from_tier, to_tier=to_tier).inc()
    else:
        _simple_metrics.inc_counter(
            "aragora_memory_tier_transitions_total",
            {"from_tier": from_tier, "to_tier": to_tier},
        )


def record_memory_operation(operation: str) -> None:
    """Record a memory operation.

    Args:
        operation: Operation type (store, retrieve, consolidate, prune)
    """
    if PROMETHEUS_AVAILABLE:
        MEMORY_OPERATIONS.labels(operation=operation).inc()
    else:
        _simple_metrics.inc_counter(
            "aragora_memory_operations_total",
            {"operation": operation},
        )


# ============================================================================
# Decorators for Easy Instrumentation
# ============================================================================


def timed_http_request(endpoint: str) -> Callable[[Callable], Callable]:
    """Decorator to time HTTP request handlers.

    Args:
        endpoint: The HTTP endpoint being timed (e.g., "/api/debates")

    Returns:
        Decorator function that wraps handlers with timing instrumentation.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                status = getattr(result, "status_code", 200) if result else 200
                return result
            except Exception:  # noqa: BLE001 - Re-raised after recording status
                status = 500
                raise
            finally:
                duration = time.perf_counter() - start
                record_http_request("GET", endpoint, status, duration)

        return wrapper

    return decorator


def timed_agent_generation(agent_type: str, model: str) -> Callable[[Callable], Callable]:
    """Decorator to time agent generation.

    Args:
        agent_type: Type of agent being timed (e.g., "anthropic-api")
        model: Model name being used (e.g., "claude-3-sonnet")

    Returns:
        Async decorator function that wraps generators with timing instrumentation.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                record_agent_failure(agent_type, type(e).__name__)
                raise
            finally:
                duration = time.perf_counter() - start
                record_agent_generation(agent_type, model, duration)

        return wrapper

    return decorator


def timed_db_query(operation: str, table: str) -> Callable[[Callable], Callable]:
    """Decorator to time database query execution.

    Args:
        operation: Query operation type (select, insert, update, delete)
        table: Table name being queried

    Returns:
        Decorator function that wraps queries with timing instrumentation.

    Usage:
        @timed_db_query("select", "debates")
        def list_debates(self, limit: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                record_db_error(type(e).__name__, operation)
                raise
            finally:
                duration = time.perf_counter() - start
                record_db_query(operation, table, duration)

        return wrapper

    return decorator


def timed_db_query_async(operation: str, table: str) -> Callable[[Callable], Callable]:
    """Async decorator to time database query execution.

    Args:
        operation: Query operation type (select, insert, update, delete)
        table: Table name being queried

    Returns:
        Async decorator function that wraps queries with timing instrumentation.

    Usage:
        @timed_db_query_async("select", "debates")
        async def list_debates(self, limit: int):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                record_db_error(type(e).__name__, operation)
                raise
            finally:
                duration = time.perf_counter() - start
                record_db_query(operation, table, duration)

        return wrapper

    return decorator


# ============================================================================
# Nomic Loop Phase Metrics
# ============================================================================


def record_nomic_phase(
    phase: str,
    outcome: str,
    duration_seconds: float,
) -> None:
    """Record a nomic loop phase execution.

    Args:
        phase: Phase name (context, debate, design, implement, verify, commit)
        outcome: Phase outcome (success, failure, skipped)
        duration_seconds: Time spent in the phase
    """
    if PROMETHEUS_AVAILABLE:
        NOMIC_PHASE_DURATION.labels(phase=phase, outcome=outcome).observe(duration_seconds)
        NOMIC_PHASE_TOTAL.labels(phase=phase, outcome=outcome).inc()
    else:
        _simple_metrics.observe_histogram(
            "aragora_nomic_phase_duration_seconds",
            duration_seconds,
            {"phase": phase, "outcome": outcome},
        )
        _simple_metrics.inc_counter(
            "aragora_nomic_phases_total",
            {"phase": phase, "outcome": outcome},
        )


def record_nomic_cycle(
    outcome: str,
    duration_seconds: float,
) -> None:
    """Record a complete nomic cycle execution.

    Args:
        outcome: Cycle outcome (success, failure, partial)
        duration_seconds: Total cycle time
    """
    if PROMETHEUS_AVAILABLE:
        NOMIC_CYCLE_DURATION.labels(outcome=outcome).observe(duration_seconds)
        NOMIC_CYCLE_TOTAL.labels(outcome=outcome).inc()
    else:
        _simple_metrics.observe_histogram(
            "aragora_nomic_cycle_duration_seconds",
            duration_seconds,
            {"outcome": outcome},
        )
        _simple_metrics.inc_counter(
            "aragora_nomic_cycles_total",
            {"outcome": outcome},
        )


def record_nomic_agent_phase(
    phase: str,
    agent: str,
    duration_seconds: float,
) -> None:
    """Record time spent by an agent in a phase.

    Args:
        phase: Phase name (context, debate, design, implement, verify)
        agent: Agent name (claude, codex, gemini, grok)
        duration_seconds: Time the agent spent in this phase
    """
    if PROMETHEUS_AVAILABLE:
        NOMIC_AGENT_PHASE_DURATION.labels(phase=phase, agent=agent).observe(duration_seconds)
    else:
        _simple_metrics.observe_histogram(
            "aragora_nomic_agent_phase_seconds",
            duration_seconds,
            {"phase": phase, "agent": agent},
        )


def timed_nomic_phase(phase: str) -> Callable[[Callable], Callable]:
    """Async decorator to time nomic phase execution.

    Args:
        phase: Phase name (context, debate, design, implement, verify, commit)

    Returns:
        Async decorator that wraps phase execution with timing.

    Usage:
        @timed_nomic_phase("debate")
        async def execute(self) -> DebateResult:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            outcome = "success"
            try:
                result = await func(*args, **kwargs)
                # Check result for success indicator
                if hasattr(result, "get") and not result.get("success", True):
                    outcome = "failure"
                return result
            except Exception:  # noqa: BLE001 - Re-raised after recording outcome
                outcome = "failure"
                raise
            finally:
                duration = time.perf_counter() - start
                record_nomic_phase(phase, outcome, duration)

        return wrapper

    return decorator


# ============================================================================
# Control Plane Metrics Recording Functions
# ============================================================================


def record_control_plane_task_submitted(task_type: str, priority: str) -> None:
    """Record a task submitted to the control plane.

    Args:
        task_type: Type of task (e.g., debate, document_processing, audit)
        priority: Task priority (low, normal, high, urgent)
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_TASKS_TOTAL.labels(task_type=task_type, priority=priority).inc()
    else:
        _simple_metrics.inc_counter(
            "aragora_control_plane_tasks_total",
            {"task_type": task_type, "priority": priority},
        )


def record_control_plane_task_status(status: str, count: int) -> None:
    """Record the count of tasks by status.

    Args:
        status: Task status (pending, running, completed, failed, cancelled)
        count: Number of tasks in this status
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_TASK_STATUS.labels(status=status).set(count)
    else:
        _simple_metrics.set_gauge(
            "aragora_control_plane_task_status_count",
            count,
            {"status": status},
        )


def record_control_plane_task_completed(
    task_type: str, outcome: str, duration_seconds: float
) -> None:
    """Record a task completion.

    Args:
        task_type: Type of task
        outcome: Task outcome (completed, failed, timeout)
        duration_seconds: Time to complete the task
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_TASK_DURATION.labels(
            task_type=task_type, outcome=outcome
        ).observe(duration_seconds)
    else:
        _simple_metrics.observe_histogram(
            "aragora_control_plane_task_duration_seconds",
            duration_seconds,
            {"task_type": task_type, "outcome": outcome},
        )


def record_control_plane_queue_depth(priority: str, depth: int) -> None:
    """Record the queue depth by priority.

    Args:
        priority: Task priority
        depth: Number of tasks in queue
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_QUEUE_DEPTH.labels(priority=priority).set(depth)
    else:
        _simple_metrics.set_gauge(
            "aragora_control_plane_queue_depth",
            depth,
            {"priority": priority},
        )


def record_control_plane_agents(status: str, count: int) -> None:
    """Record the count of agents by status.

    Args:
        status: Agent status (available, busy, offline)
        count: Number of agents in this status
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_AGENTS_REGISTERED.labels(status=status).set(count)
    else:
        _simple_metrics.set_gauge(
            "aragora_control_plane_agents_registered",
            count,
            {"status": status},
        )


def record_control_plane_agent_health(agent_id: str, health_value: int) -> None:
    """Record agent health status.

    Args:
        agent_id: Agent identifier
        health_value: Health value (0=unhealthy, 1=degraded, 2=healthy)
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_AGENT_HEALTH.labels(agent_id=agent_id).set(health_value)
    else:
        _simple_metrics.set_gauge(
            "aragora_control_plane_agent_health",
            health_value,
            {"agent_id": agent_id},
        )


def record_control_plane_agent_latency(agent_id: str, latency_seconds: float) -> None:
    """Record agent health check latency.

    Args:
        agent_id: Agent identifier
        latency_seconds: Health check latency in seconds
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_AGENT_LATENCY.labels(agent_id=agent_id).observe(latency_seconds)
    else:
        _simple_metrics.observe_histogram(
            "aragora_control_plane_agent_latency_seconds",
            latency_seconds,
            {"agent_id": agent_id},
        )


def record_control_plane_task_retry(task_type: str, reason: str) -> None:
    """Record a task retry.

    Args:
        task_type: Type of task being retried
        reason: Retry reason (timeout, error, capability_mismatch)
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_TASK_RETRIES.labels(task_type=task_type, reason=reason).inc()
    else:
        _simple_metrics.inc_counter(
            "aragora_control_plane_task_retries_total",
            {"task_type": task_type, "reason": reason},
        )


def record_control_plane_dead_letter_queue(size: int) -> None:
    """Record the size of the dead letter queue.

    Args:
        size: Number of tasks in dead letter queue
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_DEAD_LETTER_QUEUE.set(size)
    else:
        _simple_metrics.set_gauge("aragora_control_plane_dead_letter_queue_size", size)


def record_control_plane_claim_latency(priority: str, latency_seconds: float) -> None:
    """Record task claim latency.

    Args:
        priority: Task priority
        latency_seconds: Time to claim the task
    """
    if PROMETHEUS_AVAILABLE:
        CONTROL_PLANE_CLAIM_LATENCY.labels(priority=priority).observe(latency_seconds)
    else:
        _simple_metrics.observe_histogram(
            "aragora_control_plane_claim_latency_seconds",
            latency_seconds,
            {"priority": priority},
        )


# ============================================================================
# RLM (Recursive Language Models) Metrics Recording Functions
# ============================================================================


def record_rlm_compression(
    source_type: str,
    original_tokens: int,
    compressed_tokens: int,
    levels: int = 1,
    duration_seconds: float = 0.0,
    success: bool = True,
) -> None:
    """Record an RLM compression operation.

    Args:
        source_type: Type of content compressed (debate, document, knowledge)
        original_tokens: Token count before compression
        compressed_tokens: Token count after compression
        levels: Number of abstraction levels created
        duration_seconds: Time taken for compression
        success: Whether compression succeeded
    """
    status = "success" if success else "failure"

    if PROMETHEUS_AVAILABLE:
        RLM_COMPRESSIONS.labels(source_type=source_type, status=status).inc()

        if success and original_tokens > 0:
            ratio = compressed_tokens / original_tokens
            RLM_COMPRESSION_RATIO.labels(source_type=source_type).observe(ratio)

            tokens_saved = original_tokens - compressed_tokens
            if tokens_saved > 0:
                RLM_TOKENS_SAVED.labels(source_type=source_type).inc(tokens_saved)

            RLM_CONTEXT_LEVELS.labels(source_type=source_type).observe(levels)

        if duration_seconds > 0:
            RLM_COMPRESSION_DURATION.labels(
                source_type=source_type,
                levels=str(levels),
            ).observe(duration_seconds)
    else:
        _simple_metrics.inc_counter(
            "aragora_rlm_compressions_total",
            {"source_type": source_type, "status": status},
        )
        if success and original_tokens > 0:
            ratio = compressed_tokens / original_tokens
            _simple_metrics.observe_histogram(
                "aragora_rlm_compression_ratio",
                ratio,
                {"source_type": source_type},
            )
            tokens_saved = original_tokens - compressed_tokens
            if tokens_saved > 0:
                _simple_metrics.inc_counter(
                    "aragora_rlm_tokens_saved_total",
                    {"source_type": source_type},
                    tokens_saved,
                )
        if duration_seconds > 0:
            _simple_metrics.observe_histogram(
                "aragora_rlm_compression_duration_seconds",
                duration_seconds,
                {"source_type": source_type, "levels": str(levels)},
            )


def record_rlm_query(
    query_type: str,
    level: str = "SUMMARY",
    duration_seconds: float = 0.0,
) -> None:
    """Record an RLM context query.

    Args:
        query_type: Type of query (drill_down, roll_up, search, etc.)
        level: Abstraction level queried (ABSTRACT, SUMMARY, DETAILED, FULL)
        duration_seconds: Time taken for query
    """
    if PROMETHEUS_AVAILABLE:
        RLM_QUERIES.labels(query_type=query_type, level=level).inc()
        if duration_seconds > 0:
            RLM_QUERY_DURATION.labels(query_type=query_type).observe(duration_seconds)
    else:
        _simple_metrics.inc_counter(
            "aragora_rlm_queries_total",
            {"query_type": query_type, "level": level},
        )
        if duration_seconds > 0:
            _simple_metrics.observe_histogram(
                "aragora_rlm_query_duration_seconds",
                duration_seconds,
                {"query_type": query_type},
            )


def record_rlm_cache_hit() -> None:
    """Record an RLM compression cache hit."""
    if PROMETHEUS_AVAILABLE:
        RLM_CACHE_HITS.inc()
    else:
        _simple_metrics.inc_counter("aragora_rlm_cache_hits_total")


def record_rlm_cache_miss() -> None:
    """Record an RLM compression cache miss."""
    if PROMETHEUS_AVAILABLE:
        RLM_CACHE_MISSES.inc()
    else:
        _simple_metrics.inc_counter("aragora_rlm_cache_misses_total")


def set_rlm_memory_usage(bytes_used: int) -> None:
    """Set current memory usage for RLM context cache.

    Args:
        bytes_used: Memory usage in bytes
    """
    if PROMETHEUS_AVAILABLE:
        RLM_MEMORY_USAGE.set(bytes_used)
    else:
        _simple_metrics.set_gauge("aragora_rlm_memory_bytes", bytes_used)


def record_rlm_refinement(
    strategy: str,
    iterations: int,
    success: bool,
    duration_seconds: float = 0.0,
) -> None:
    """Record an RLM iterative refinement operation.

    Args:
        strategy: Decomposition strategy used (auto, grep, partition_map, etc.)
        iterations: Number of iterations until ready=True (or max iterations)
        success: Whether ready=True was achieved before max iterations
        duration_seconds: Total time for refinement loop
    """
    if PROMETHEUS_AVAILABLE:
        RLM_REFINEMENT_ITERATIONS.labels(strategy=strategy).observe(iterations)
        if success:
            RLM_REFINEMENT_SUCCESS.labels(strategy=strategy).inc()
        if duration_seconds > 0:
            RLM_REFINEMENT_DURATION.labels(strategy=strategy).observe(duration_seconds)
    else:
        _simple_metrics.observe_histogram(
            "aragora_rlm_refinement_iterations",
            iterations,
            {"strategy": strategy},
        )
        if success:
            _simple_metrics.inc_counter(
                "aragora_rlm_refinement_success_total",
                {"strategy": strategy},
            )
        if duration_seconds > 0:
            _simple_metrics.observe_histogram(
                "aragora_rlm_refinement_duration_seconds",
                duration_seconds,
                {"strategy": strategy},
            )


def record_rlm_ready_false(iteration: int) -> None:
    """Record when LLM signals ready=False (needs refinement).

    Args:
        iteration: Current iteration number (0-indexed)
    """
    if PROMETHEUS_AVAILABLE:
        RLM_READY_FALSE_RATE.labels(iteration=str(iteration)).inc()
    else:
        _simple_metrics.inc_counter(
            "aragora_rlm_ready_false_total",
            {"iteration": str(iteration)},
        )


def timed_rlm_compression(source_type: str) -> Callable[[Callable], Callable]:
    """Async decorator to time RLM compression operations.

    Args:
        source_type: Type of content being compressed (debate, document, knowledge)

    Returns:
        Async decorator that wraps compression with timing.

    Usage:
        @timed_rlm_compression("debate")
        async def compress_debate(self, debate: DebateResult) -> RLMContext:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            success = True
            original_tokens = 0
            compressed_tokens = 0
            levels = 1
            try:
                result = await func(*args, **kwargs)
                # Try to extract metrics from result
                if hasattr(result, "original_tokens"):
                    original_tokens = result.original_tokens
                if hasattr(result, "compressed_tokens"):
                    compressed_tokens = result.compressed_tokens
                if hasattr(result, "levels"):
                    levels = len(result.levels) if hasattr(result.levels, "__len__") else result.levels
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = time.perf_counter() - start
                record_rlm_compression(
                    source_type=source_type,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    levels=levels,
                    duration_seconds=duration,
                    success=success,
                )

        return wrapper

    return decorator


def timed_rlm_refinement(strategy: str = "auto") -> Callable[[Callable], Callable]:
    """Async decorator to time RLM refinement operations.

    Args:
        strategy: Decomposition strategy being used

    Returns:
        Async decorator that wraps refinement with timing.

    Usage:
        @timed_rlm_refinement("grep")
        async def query_with_refinement(self, query: str) -> RLMResult:
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            iterations = 1
            success = False
            try:
                result = await func(*args, **kwargs)
                # Try to extract metrics from result
                if hasattr(result, "iteration"):
                    iterations = result.iteration + 1
                if hasattr(result, "ready"):
                    success = result.ready
                return result
            finally:
                duration = time.perf_counter() - start
                record_rlm_refinement(
                    strategy=strategy,
                    iterations=iterations,
                    success=success,
                    duration_seconds=duration,
                )

        return wrapper

    return decorator
