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
from typing import Dict, Optional, Callable
from functools import wraps

# Try to import prometheus_client, fall back to simple implementation
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
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
            except Exception as e:
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
            except Exception as e:
                outcome = "failure"
                raise
            finally:
                duration = time.perf_counter() - start
                record_nomic_phase(phase, outcome, duration)

        return wrapper

    return decorator
