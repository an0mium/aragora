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

import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Dict

logger = logging.getLogger(__name__)

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

    # Cost metrics
    COST_USD_TOTAL = Counter(
        "aragora_cost_usd_total",
        "Total cost in USD (multiply by 1e-6 for actual dollars)",
        ["provider", "model", "agent_id"],
    )

    COST_PER_DEBATE = Histogram(
        "aragora_debate_cost_usd",
        "Cost per debate in USD (multiply by 1e-6 for actual dollars)",
        ["provider"],
        buckets=[1000, 5000, 10000, 50000, 100000, 500000, 1000000],  # micro-dollars
    )

    BUDGET_UTILIZATION = Gauge(
        "aragora_budget_utilization_percent",
        "Current budget utilization percentage",
        ["workspace_id", "budget_type"],  # budget_type: daily, monthly, per_debate
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

    # Deliberation metrics
    DELIBERATION_DURATION = Histogram(
        "aragora_deliberation_duration_seconds",
        "Deliberation execution time",
        ["status", "consensus_reached"],
        buckets=[10, 30, 60, 120, 300, 600, 900, 1800],
    )

    DELIBERATION_SLA_COMPLIANCE = Counter(
        "aragora_deliberation_sla_total",
        "SLA compliance counts",
        ["level"],  # compliant, warning, critical, violated
    )

    DELIBERATION_CONSENSUS_CONFIDENCE = Histogram(
        "aragora_deliberation_consensus_confidence",
        "Distribution of consensus confidence scores",
        buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    )

    DELIBERATION_ROUNDS = Histogram(
        "aragora_deliberation_rounds_total",
        "Number of rounds per deliberation",
        ["outcome"],
        buckets=[1, 2, 3, 4, 5, 7, 10, 15],
    )

    DELIBERATION_AGENTS = Histogram(
        "aragora_deliberation_agents_count",
        "Number of agents per deliberation",
        buckets=[2, 3, 4, 5, 6, 8, 10],
    )

    DELIBERATION_TOTAL = Counter(
        "aragora_deliberation_total",
        "Total deliberations",
        ["outcome"],  # consensus, no_consensus, failed, timeout
    )

    AGENT_UTILIZATION = Gauge(
        "aragora_agent_utilization_ratio",
        "Agent busy vs available ratio",
        ["agent_id"],
    )

    POLICY_DECISIONS = Counter(
        "aragora_policy_decisions_total",
        "Policy evaluation decisions",
        ["decision", "policy_type"],  # decision: allow, deny, warn
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

    # Knowledge Mound metrics
    KNOWLEDGE_VISIBILITY_CHANGES = Counter(
        "aragora_knowledge_visibility_changes_total",
        "Visibility level changes on knowledge items",
        ["from_level", "to_level", "workspace_id"],
    )

    KNOWLEDGE_ACCESS_GRANTS = Counter(
        "aragora_knowledge_access_grants_total",
        "Access grants created or revoked",
        ["action", "grantee_type", "workspace_id"],  # action: grant/revoke
    )

    KNOWLEDGE_SHARES = Counter(
        "aragora_knowledge_shares_total",
        "Knowledge sharing operations",
        ["action", "target_type"],  # action: share/accept/decline/revoke
    )

    KNOWLEDGE_SHARED_ITEMS = Gauge(
        "aragora_knowledge_shared_items_count",
        "Shared items pending acceptance per workspace",
        ["workspace_id"],
    )

    KNOWLEDGE_GLOBAL_FACTS = Counter(
        "aragora_knowledge_global_facts_total",
        "Global/verified facts stored or promoted",
        ["action"],  # action: stored/promoted/queried
    )

    KNOWLEDGE_GLOBAL_QUERIES = Counter(
        "aragora_knowledge_global_queries_total",
        "Queries against global knowledge",
        ["has_results"],  # has_results: true/false
    )

    KNOWLEDGE_FEDERATION_SYNCS = Counter(
        "aragora_knowledge_federation_syncs_total",
        "Federation sync operations",
        ["region_id", "direction", "status"],  # direction: push/pull
    )

    KNOWLEDGE_FEDERATION_NODES = Counter(
        "aragora_knowledge_federation_nodes_total",
        "Nodes synced via federation",
        ["region_id", "direction"],
    )

    KNOWLEDGE_FEDERATION_LATENCY = Histogram(
        "aragora_knowledge_federation_latency_seconds",
        "Federation sync operation latency",
        ["region_id", "direction"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )

    KNOWLEDGE_FEDERATION_REGIONS = Gauge(
        "aragora_knowledge_federation_regions_count",
        "Federated regions by status",
        ["status"],  # status: enabled/disabled/healthy/unhealthy
    )

    # ========================================================================
    # Cross-Pollination Event Metrics
    # ========================================================================

    CROSS_POLL_EVENTS_TOTAL = Counter(
        "aragora_cross_pollination_events_total",
        "Total cross-pollination events dispatched",
        ["event_type"],
    )

    CROSS_POLL_HANDLER_CALLS = Counter(
        "aragora_cross_pollination_handler_calls_total",
        "Total handler invocations",
        ["handler", "status"],  # status: success, failure, skipped
    )

    CROSS_POLL_HANDLER_DURATION = Histogram(
        "aragora_cross_pollination_handler_duration_seconds",
        "Handler execution time",
        ["handler"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
    )

    CROSS_POLL_CIRCUIT_BREAKER = Gauge(
        "aragora_cross_pollination_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open)",
        ["handler"],
    )

    CROSS_POLL_SUBSCRIBERS = Gauge(
        "aragora_cross_pollination_subscribers",
        "Number of registered subscribers",
        ["event_type"],
    )

    # ========================================================================
    # Knowledge Mound Bidirectional Flow Metrics
    # ========================================================================

    KM_INBOUND_EVENTS = Counter(
        "aragora_km_inbound_events_total",
        "Events flowing INTO Knowledge Mound from other subsystems",
        ["source", "event_type"],  # source: memory, belief, rlm, ranking, insights, etc.
    )

    KM_OUTBOUND_EVENTS = Counter(
        "aragora_km_outbound_events_total",
        "Events flowing OUT of Knowledge Mound to other subsystems",
        ["target", "event_type"],  # target: memory, belief, debate, trickster, etc.
    )

    KM_ADAPTER_SYNC = Counter(
        "aragora_km_adapter_sync_total",
        "Adapter sync operations to/from Knowledge Mound",
        ["adapter", "direction", "status"],  # direction: to_mound, from_mound
    )

    KM_ADAPTER_SYNC_DURATION = Histogram(
        "aragora_km_adapter_sync_duration_seconds",
        "Adapter sync operation duration",
        ["adapter", "direction"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30],
    )

    KM_STALENESS_CHECKS = Counter(
        "aragora_km_staleness_checks_total",
        "Staleness check operations",
        ["workspace", "status"],  # status: completed, failed, skipped
    )

    KM_STALE_NODES_FOUND = Gauge(
        "aragora_km_stale_nodes_found",
        "Number of stale nodes found in last check",
        ["workspace"],
    )

    KM_NODES_BY_SOURCE = Gauge(
        "aragora_km_nodes_by_source",
        "Knowledge nodes by source type",
        ["source"],
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


def record_cost_usd(provider: str, model: str, agent_id: str, cost_usd: float) -> None:
    """Record cost in USD.

    Cost is stored as micro-dollars (1e-6) for precision with counters.
    """
    micro_dollars = int(cost_usd * 1_000_000)
    if PROMETHEUS_AVAILABLE:
        COST_USD_TOTAL.labels(provider=provider, model=model, agent_id=agent_id).inc(micro_dollars)
    else:
        _simple_metrics.inc_counter(
            "aragora_cost_usd_total",
            {"provider": provider, "model": model, "agent_id": agent_id},
            micro_dollars,
        )


def record_debate_cost(provider: str, cost_usd: float) -> None:
    """Record cost for a completed debate."""
    micro_dollars = int(cost_usd * 1_000_000)
    if PROMETHEUS_AVAILABLE:
        COST_PER_DEBATE.labels(provider=provider).observe(micro_dollars)
    else:
        _simple_metrics.observe_histogram(
            "aragora_debate_cost_usd",
            micro_dollars,
            {"provider": provider},
        )


def set_budget_utilization(workspace_id: str, budget_type: str, utilization_percent: float) -> None:
    """Set current budget utilization percentage."""
    if PROMETHEUS_AVAILABLE:
        BUDGET_UTILIZATION.labels(workspace_id=workspace_id, budget_type=budget_type).set(
            utilization_percent
        )
    else:
        _simple_metrics.set_gauge(
            "aragora_budget_utilization_percent",
            utilization_percent,
            {"workspace_id": workspace_id, "budget_type": budget_type},
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


def initialize_circuit_breaker_metrics() -> None:
    """Initialize circuit breaker metrics integration.

    Registers a callback with the resilience module to automatically
    export circuit breaker state changes to Prometheus.

    Call this once during server startup.
    """
    try:
        from aragora.resilience import set_metrics_callback

        set_metrics_callback(set_circuit_breaker_state)
        logger.info("Circuit breaker metrics integration initialized")
    except ImportError:
        logger.debug("Resilience module not available for metrics integration")


def export_circuit_breaker_metrics() -> None:
    """Export all circuit breaker states to Prometheus.

    Call this periodically (e.g., every 30s) to ensure metrics
    are up-to-date even if state changes were missed.
    """
    try:
        from aragora.resilience import get_circuit_breaker_metrics

        metrics = get_circuit_breaker_metrics()
        for name, cb_data in metrics.get("circuit_breakers", {}).items():
            status = cb_data.get("status", "closed")
            state_value = {"closed": 0, "open": 1, "half-open": 2}.get(status, 0)
            set_circuit_breaker_state(name, state_value)

            # Also export entity-level states if in multi-entity mode
            entity_mode = cb_data.get("entity_mode", {})
            for entity in entity_mode.get("open_entities", []):
                set_circuit_breaker_state(f"{name}:{entity}", 1)  # open

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error exporting circuit breaker metrics: {e}")


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
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError) as e:
                logger.warning("HTTP request to %s failed: %s", endpoint, e)
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
            except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError, OSError) as e:
                logger.warning("Agent %s generation failed: %s", agent_type, e)
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
            except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError, OSError) as e:
                logger.warning("DB %s on %s failed: %s", operation, table, e)
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
            except (ValueError, TypeError, KeyError, RuntimeError, TimeoutError, OSError) as e:
                logger.warning("Async DB %s on %s failed: %s", operation, table, e)
                record_db_error(type(e).__name__, operation)
                raise
            finally:
                duration = time.perf_counter() - start
                record_db_query(operation, table, duration)

        return wrapper

    return decorator


# ============================================================================
# Extracted modules (import directly to avoid circular imports)
# ============================================================================
# Nomic metrics: from aragora.server.prometheus_nomic import record_nomic_phase, ...
# Control Plane metrics: from aragora.server.prometheus_control_plane import ...
# RLM metrics: from aragora.server.prometheus_rlm import record_rlm_compression, ...
# Knowledge metrics: from aragora.server.prometheus_knowledge import ...

__all__ = [
    # Core
    "PROMETHEUS_AVAILABLE",
    "get_metrics_output",
    "is_prometheus_available",
    "get_prometheus_metrics",
    # Core recording functions
    "record_debate_completed",
    "record_tokens_used",
    "record_agent_generation",
    "record_agent_failure",
    "set_circuit_breaker_state",
    "initialize_circuit_breaker_metrics",
    "export_circuit_breaker_metrics",
    "record_http_request",
    "set_websocket_connections",
    "record_websocket_message",
    "record_rate_limit_hit",
    "set_rate_limit_tokens_tracked",
    "set_cache_size",
    "record_cache_hit",
    "record_cache_miss",
    "set_server_info",
    "record_db_query",
    "record_db_error",
    "set_db_pool_size",
    "set_memory_tier_size",
    "record_memory_tier_transition",
    "record_memory_operation",
    # Decorators
    "timed_http_request",
    "timed_agent_generation",
    "timed_db_query",
    "timed_db_query_async",
    # Note: Nomic, Control Plane, RLM, and Knowledge metrics are in extracted modules.
    # Import directly from: prometheus_nomic, prometheus_control_plane,
    # prometheus_rlm, prometheus_knowledge
]
